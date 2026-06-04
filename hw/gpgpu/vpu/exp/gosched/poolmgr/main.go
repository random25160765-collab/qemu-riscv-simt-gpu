package poolmgr

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

func Main() {
	numVPU := flag.Int("n", 4, "number of VPU instances")
	vramMB := flag.Int("vram", 64, "VRAM size in MB per VPU")
	daemonPath := flag.String("daemon", "./vpu", "path to VPU daemon binary")
	listen := flag.String("listen", ":8080", "HTTP listen address")
	flag.Parse()

	log.Printf("poolmgr: starting %d VPU instances (%dMB each), daemon=%s",
		*numVPU, *vramMB, *daemonPath)

	// 1. Create VPU instances
	instances := make([]*VPUInstance, *numVPU)
	var wg sync.WaitGroup
	errCh := make(chan error, *numVPU)

	for i := 0; i < *numVPU; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			v, err := NewVPU(id, *vramMB, *daemonPath)
			if err != nil {
				errCh <- fmt.Errorf("vpu %d: %w", id, err)
				return
			}
			instances[id] = v
			log.Printf("vpu %d: ready", id)
		}(i)
	}
	wg.Wait()
	close(errCh)

	// check for creation errors
	errCount := 0
	for e := range errCh {
		log.Printf("ERROR: %v", e)
		errCount++
	}
	if errCount == *numVPU {
		log.Fatal("all VPU instances failed to start")
	}

	// filter out nil instances
	valid := instances[:0]
	for _, v := range instances {
		if v != nil { valid = append(valid, v) }
	}
	instances = valid
	log.Printf("poolmgr: %d/%d VPU instances ready", len(instances), *numVPU)

	// 2. Create allocator
	alloc := NewAllocator(instances)
	defer alloc.Close()

	// 3. Health check goroutine
	go func() {
		for range time.Tick(5 * time.Second) {
			stats := alloc.Stats()
			log.Printf("health: %+v", stats)
		}
	}()

	// 4. HTTP API
	mux := http.NewServeMux()

	// POST /gpu/alloc — acquire a VPU, returns {vpu_id}
	mux.HandleFunc("/gpu/alloc", func(w http.ResponseWriter, r *http.Request) {
		v, err := alloc.Alloc()
		if err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"vpu_id": v.ID,
			"status": "allocated",
		})
	})

	// POST /gpu/free — release a VPU
	mux.HandleFunc("/gpu/free", func(w http.ResponseWriter, r *http.Request) {
		var req struct{ VPUID int `json:"vpu_id"` }
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if req.VPUID < 0 || req.VPUID >= len(instances) {
			http.Error(w, "invalid vpu_id", http.StatusBadRequest)
			return
		}
		alloc.Free(instances[req.VPUID])
		json.NewEncoder(w).Encode(map[string]string{"status": "freed"})
	})

	// POST /gpu/dispatch — dispatch kernel to a specific VPU
	mux.HandleFunc("/gpu/dispatch", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			VPUID    int      `json:"vpu_id"`
			KernAddr uint32   `json:"kern_addr"`
			Grid     [3]uint32 `json:"grid"`
			Block    [3]uint32 `json:"block"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if req.VPUID < 0 || req.VPUID >= len(instances) {
			http.Error(w, "invalid vpu_id", http.StatusBadRequest)
			return
		}
		v := instances[req.VPUID]
		// kernel loaded separately — dispatch reuses what's in VRAM
		ret, err := v.Dispatch(nil, req.KernAddr, req.Grid, req.Block)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"ret": ret,
		})
	})

	// GET /metrics — Prometheus-style stats
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		s := alloc.Stats()
		fmt.Fprintf(w, "vpu_total %d\n", s.Total)
		fmt.Fprintf(w, "vpu_idle %d\n", s.Idle)
		fmt.Fprintf(w, "vpu_busy %d\n", s.Busy)
		fmt.Fprintf(w, "vpu_error %d\n", s.Error)
		fmt.Fprintf(w, "vpu_dispatches %d\n", s.Dispatch)
	})

	// GET / — status page
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		s := alloc.Stats()
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, `<html><body>
<h1>VPU Pool Manager</h1>
<p>Instances: %d total, %d idle, %d busy, %d error</p>
<p>Dispatches: %d</p>
<p><a href="/debug/pprof/">pprof</a> | <a href="/metrics">metrics</a></p>
</body></html>`, s.Total, s.Idle, s.Busy, s.Error, s.Dispatch)
	})

	// 5. Start server
	server := &http.Server{Addr: *listen, Handler: mux}
	go func() {
		log.Printf("listening on %s", *listen)
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	// 6. Wait for shutdown
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Println("shutting down...")
	server.Close()
}
