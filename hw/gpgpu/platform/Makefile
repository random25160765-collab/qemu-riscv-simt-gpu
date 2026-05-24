probe-web:
	@fuser -k 8080/tcp 2>/dev/null; sleep 0.2
	PYTHONUNBUFFERED=1 python3 backend/orchestrate/main.py

test-backend:
	@fuser -k 18080/tcp 2>/dev/null; fuser -k 18081/tcp 2>/dev/null; sleep 0.2
	@echo "=== 后端单元测试 (parse_frame) ==="
	@python3 backend/test_pipeline.py
	@echo
	@echo "=== 后端集成测试 (raw_bus → bus → SSE) ==="
	@python3 backend/test_pipeline.py --full

test: test-backend
	@echo
	@echo "前端测试: 浏览器打开 http://localhost:8080/test"

.PHONY: probe-web test-backend test
