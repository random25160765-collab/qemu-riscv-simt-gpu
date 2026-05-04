# Conflux 运行时库 API 参考

本文档描述了用于 Conflux GPGPU 模拟设备的运行时库全部公共接口。库被组织为多个模块，彼此独立，可组合使用。

---

## 1. 错误码 (`conflux_error.h`)

所有函数通过返回 `conflux_error_t` 或通过输出参数报告错误。错误码为负整数，成功返回 `CONFLUX_SUCCESS` (0)。

| 错误码 | 值 | 说明 |
|--------|----|------|
| `CONFLUX_SUCCESS` | 0 | 成功 |
| `CONFLUX_ERR_INVALID` | -1 | 参数无效（NULL、越界等） |
| `CONFLUX_ERR_NOMEM` | -2 | 主机内存不足 |
| `CONFLUX_ERR_TIMEOUT` | -3 | 操作超时 |
| `CONFLUX_ERR_BUSY` | -4 | 设备忙 |
| `CONFLUX_ERR_MEM_OUT_OF_DEVICE` | -10 | 设备显存不足 |
| `CONFLUX_ERR_MEM_FRAGMENTED` | -11 | 显存碎片导致分配失败 |
| `CONFLUX_ERR_MEM_INVALID_ADDR` | -12 | 设备地址无效 |
| `CONFLUX_ERR_MEM_OVERLAP` | -13 | 地址重叠 |
| `CONFLUX_ERR_EVENT_INVALID_STATE` | -20 | 事件状态转换不合法 |
| `CONFLUX_ERR_EVENT_NULL` | -21 | 事件指针为空 |
| `CONFLUX_ERR_DEVICE_NOT_READY` | -30 | 设备未初始化 |
| `CONFLUX_ERR_DEVICE_FAULT` | -31 | 设备硬件错误 |
| `CONFLUX_ERR_DEVICE_RESET_FAIL` | -32 | 设备复位失败 |
| `CONFLUX_ERR_CMD_INVALID_OPCODE` | -40 | 不支持的命令码 |
| `CONFLUX_ERR_CMD_QUEUE_FULL` | -41 | 命令队列已满 |

**辅助函数**

```c
const char *conflux_strerror(int err);
```
返回错误码对应的可读字符串。

---

## 2. 位图显存分配器 (`conflux_allocator.h`)

管理设备全局内存的分配与释放，采用固定块大小的位图算法。

### 数据结构

```c
typedef struct {
    uint64_t base_addr;
    uint64_t total_size;
    uint32_t block_size;
    uint32_t total_blocks;
    bitmap_word_t *bitmap;
    uint32_t free_blocks;
} conflux_allocator_t;
```

### API

```c
int conflux_allocator_init(conflux_allocator_t *alloc,
                          uint64_t base_addr,
                          uint64_t total_size,
                          uint32_t block_size);
```
初始化分配器。`block_size` 必须是 2 的幂。返回 `0` 成功，负值错误。

```c
void conflux_allocator_destroy(conflux_allocator_t *alloc);
```
释放分配器内部资源（位图内存）。

```c
uint64_t conflux_allocator_alloc(conflux_allocator_t *alloc, size_t size);
```
分配连续设备内存，返回设备物理地址。失败返回 `UINT64_MAX`。可通过 `conflux_allocator_alloc_ext`（如有）获得详细错误码。

```c
int conflux_allocator_free(conflux_allocator_t *alloc,
                          uint64_t dev_addr, size_t size);
```
释放之前分配的地址段。返回 `0` 成功，负值错误。

```c
void conflux_allocator_dump(const conflux_allocator_t *alloc);
```
打印分配器状态到标准输出。

---

## 3. 事件对象 (`conflux_event.h`)

用于异步命令的状态跟踪与同步。

### 数据结构

```c
typedef enum {
    CONFLUX_EVENT_QUEUED,
    CONFLUX_EVENT_SUBMITTED,
    CONFLUX_EVENT_RUNNING,
    CONFLUX_EVENT_COMPLETE,
    CONFLUX_EVENT_FAILED
} conflux_event_status_t;

typedef struct _conflux_event { ... } conflux_event_t;
```

### API

```c
conflux_event_t *conflux_event_create(void);
```
创建事件对象，初始状态为 `QUEUED`，引用计数 1。失败返回 `NULL`。

```c
void conflux_event_destroy(conflux_event_t *event);
```
立即销毁事件（忽略引用计数）。通常应使用 `conflux_event_release`。

```c
void conflux_event_set_queued(conflux_event_t *event);
void conflux_event_set_submitted(conflux_event_t *event);
void conflux_event_set_running(conflux_event_t *event);
void conflux_event_set_complete(conflux_event_t *event);
void conflux_event_set_failed(conflux_event_t *event, int error_code);
```
改变事件状态。`set_complete` 和 `set_failed` 会唤醒所有等待者并触发回调（若有）。

```c
conflux_event_status_t conflux_event_get_status(conflux_event_t *event);
int conflux_event_is_complete(conflux_event_t *event);
```
查询状态。`is_complete` 对 `COMPLETE` 或 `FAILED` 返回非 0。

```c
conflux_error_t conflux_event_wait(conflux_event_t *event, uint64_t timeout_ns);
```
阻塞等待事件完成，`timeout_ns=0` 表示无限等待。返回 `CONFLUX_SUCCESS`, `CONFLUX_ERR_TIMEOUT`, 或 `CONFLUX_ERR_INVALID`。

```c
void conflux_event_retain(conflux_event_t *event);
void conflux_event_release(conflux_event_t *event);
```
引用计数操作。当计数降为 0 时自动销毁事件。

```c
void conflux_event_set_callback(conflux_event_t *event,
                               void (*callback)(conflux_event_t *, void *),
                               void *data);
```
设置完成回调。事件变为 `COMPLETE` 或 `FAILED` 时调用。

```c
void conflux_event_dump(const conflux_event_t *event, char *buf, size_t buf_size);
```
格式化事件状态到字符串。

```c
uint64_t conflux_get_time_ns(void);
```
返回单调时钟当前值（纳秒），用于时间戳。

---

## 4. 命令队列 (`conflux_queue.h`)

多生产者单消费者环形队列，管理命令的异步提交与执行。

### 数据结构

```c
typedef enum {
    CONFLUX_CMD_NOP,
    CONFLUX_CMD_COPY,
    CONFLUX_CMD_KERNEL,
    CONFLUX_CMD_ALLOC,
    CONFLUX_CMD_FREE,
    CONFLUX_CMD_BARRIER,
} conflux_cmd_type_t;

typedef struct {
    conflux_cmd_type_t type;
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t size;
    uint32_t flags;
    uint32_t kernel_id;
    uint32_t reserved;
} conflux_cmd_t;

typedef struct { ... } conflux_queue_t;
```

### API

```c
conflux_queue_t *conflux_queue_create(uint32_t ring_size,
                                     int (*execute_cmd)(conflux_cmd_t *, void *),
                                     void *device_data);
```
创建队列。`ring_size` 必须是 2 的幂。`execute_cmd` 是消费者的命令处理回调。失败返回 `NULL`。

```c
void conflux_queue_destroy(conflux_queue_t *queue);
```
停止消费者并销毁队列。

```c
int conflux_queue_submit(conflux_queue_t *queue,
                        conflux_cmd_t *cmd,
                        conflux_event_t **event_out);
```
非阻塞提交命令。若 `event_out` 非空，将新创建的事件通过它返回；否则事件被内部释放。返回 `CONFLUX_SUCCESS` 或错误码。

```c
int conflux_queue_submit_sync(conflux_queue_t *queue, conflux_cmd_t *cmd);
```
提交命令并阻塞等待执行完成。返回 `CONFLUX_SUCCESS` 或错误码。

```c
int conflux_queue_drain(conflux_queue_t *queue);
```
阻塞直到所有已提交命令执行完毕。

```c
int conflux_queue_start_consumer(conflux_queue_t *queue);
int conflux_queue_stop_consumer(conflux_queue_t *queue);
```
启动/停止内部消费者线程。

```c
int conflux_queue_is_empty(const conflux_queue_t *queue);
int conflux_queue_is_full(const conflux_queue_t *queue);
uint32_t conflux_queue_pending_count(const conflux_queue_t *queue);
```
查询队列状态。

```c
void conflux_queue_dump(const conflux_queue_t *queue);
```
打印队列状态和环形缓冲区内容。

---

## 5. 内核对象 (`conflux_kernel.h`)

封装编译好的 GPU 内核及其参数、NDRange 配置。

### 数据结构

```c
typedef struct {
    char name[256];
    uint32_t kernel_id;
    void *binary;
    size_t binary_size;
    uint64_t binary_device_addr;
    int binary_uploaded;
    uint32_t num_args;
    conflux_kernel_arg_t args[CONFLUX_KERNEL_MAX_ARGS];
    uint32_t work_dim;
    size_t global_size[3];
    size_t local_size[3];
    void *queue;
} conflux_kernel_t;
```

### API

```c
conflux_kernel_t *conflux_kernel_create(const char *name,
                                       uint32_t kernel_id,
                                       void *binary,
                                       size_t binary_size);
```
创建内核对象。`binary` 指向主机端指令，可随后上传。

```c
void conflux_kernel_destroy(conflux_kernel_t *kernel);
```
销毁内核对象。

```c
int conflux_kernel_upload(conflux_kernel_t *kernel, uint64_t device_addr);
```
设置指令在设备上的地址，标记已上传。通常由命令构建器自动调用。

```c
int conflux_kernel_set_arg(conflux_kernel_t *kernel,
                          uint32_t arg_index,
                          size_t arg_size,
                          const void *arg_value,
                          int is_local);
```
设置内核参数。`is_local` 为 1 表示局部内存（`arg_value` 可为 `NULL`）。

```c
int conflux_kernel_set_work_dim(conflux_kernel_t *kernel, uint32_t work_dim);
int conflux_kernel_set_global_size(conflux_kernel_t *kernel, uint32_t dim, size_t size);
int conflux_kernel_set_local_size(conflux_kernel_t *kernel, uint32_t dim, size_t size);
```
配置 NDRange。

```c
void conflux_kernel_set_queue(conflux_kernel_t *kernel, void *queue);
```
关联命令队列（由构建器使用）。

```c
int conflux_kernel_pack_cmd(const conflux_kernel_t *kernel, conflux_cmd_t *cmd_out);
```
将内核信息打包为 `CONFLUX_CMD_KERNEL` 命令描述符。

```c
void conflux_kernel_dump(const conflux_kernel_t *kernel);
```
打印内核详情。

---

## 6. 命令构建器 (`conflux_cmd_builder.h`)

编排底层操作，提供类似 OpenCL enqueue 风格的高级接口。

### 数据结构

```c
typedef struct {
    uint64_t src_addr;
    uint64_t dst_addr;
    size_t   size;
} conflux_copy_request_t;

typedef struct {
    conflux_kernel_t *kernel;
    uint32_t work_dim;
    size_t global_size[3];
    size_t local_size[3];
} conflux_launch_request_t;

typedef struct { ... } conflux_cmd_builder_t;
```

### API

```c
conflux_cmd_builder_t *conflux_cmd_builder_create(conflux_queue_t *queue,
                                                 conflux_allocator_t *allocator);
```
创建构建器，绑定队列和分配器。

```c
void conflux_cmd_builder_destroy(conflux_cmd_builder_t *builder);
```
销毁构建器。

```c
int conflux_cmd_builder_copy(conflux_cmd_builder_t *builder,
                            const conflux_copy_request_t *req,
                            conflux_event_t **event_out);
```
提交设备内拷贝命令。

```c
int conflux_cmd_builder_launch(conflux_cmd_builder_t *builder,
                              const conflux_launch_request_t *req,
                              conflux_event_t **event_out);
```
提交内核执行命令。自动调用 `pack_cmd`。

```c
int conflux_cmd_builder_alloc(conflux_cmd_builder_t *builder,
                             size_t size,
                             uint64_t *dev_addr_out,
                             conflux_event_t **event_out);
```
从分配器分配设备内存并提交 `ALLOC` 命令。

```c
int conflux_cmd_builder_free(conflux_cmd_builder_t *builder,
                            uint64_t dev_addr,
                            size_t size,
                            conflux_event_t **event_out);
```
释放设备内存并提交 `FREE` 命令。

```c
int conflux_cmd_builder_barrier(conflux_cmd_builder_t *builder,
                               conflux_event_t **event_out);
```
提交同步屏障命令。

```c
int conflux_cmd_builder_upload_kernel(conflux_cmd_builder_t *builder,
                                     conflux_kernel_t *kernel,
                                     uint64_t *dev_addr_out);
```
从分配器获取设备地址，提交拷贝命令将内核二进制上传，并设置内核对象状态。

```c
void conflux_cmd_builder_dump(const conflux_cmd_builder_t *builder);
```
打印构建器统计信息。

---

## 7. 设备上下文 (`conflux_device.h`)

代表一个物理/虚拟 Conflux 设备实例，包含属性和显存分配器。

### 数据结构

```c
typedef struct {
    int fd;
    void *regs;
    uint64_t mmio_base;
    uint64_t mmio_size;
    char name[64];
    char vendor[32];
    uint32_t vendor_id, device_id;
    uint32_t max_compute_units;
    uint32_t max_work_item_dims;
    size_t max_work_item_sizes[3];
    size_t max_work_group_size;
    uint32_t max_clock_frequency;
    uint32_t address_bits;
    uint64_t global_mem_size;
    uint64_t local_mem_size;
    uint64_t max_mem_alloc_size;
    uint32_t mem_block_size;
    conflux_allocator_t allocator;
    uint32_t num_queues, max_queues;
    volatile uint32_t flags;
    int last_error;
    pthread_mutex_t lock;
} conflux_device_t;
```

### API

```c
conflux_device_t *conflux_device_create(void);
```
创建未初始化的设备对象，填充默认属性。

```c
void conflux_device_destroy(conflux_device_t *dev);
```
销毁设备及内部资源。

```c
int conflux_device_init(conflux_device_t *dev,
                       const char *dev_path,
                       uint64_t mmio_base,
                       uint64_t mmio_size);
```
初始化设备（打开设备文件或进入模拟模式），并初始化内部显存分配器。

```c
int conflux_device_online(conflux_device_t *dev);
int conflux_device_offline(conflux_device_t *dev);
int conflux_device_is_online(const conflux_device_t *dev);
```
上线/离线控制。

```c
int conflux_device_reset(conflux_device_t *dev);
```
复位设备状态，重新初始化分配器。

```c
void conflux_device_query_info(const conflux_device_t *dev,
                              char *buf, size_t buf_size);
```
获取设备属性字符串。

```c
conflux_allocator_t *conflux_device_get_allocator(conflux_device_t *dev);
```
返回内部分配器指针。

```c
int conflux_device_get_last_error(const conflux_device_t *dev);
```
返回最近错误码。

```c
void conflux_device_dump(const conflux_device_t *dev);
```
打印设备详细信息。

---

## 8. 平台管理 (`conflux_platform.h`)

全局单例，管理多个 Conflux 设备。

### 数据结构

```c
typedef struct {
    int dev_index;
    char path[256];
    uint64_t mmio_base, mmio_size;
    uint32_t vendor_id, device_id;
    int available;
} conflux_device_desc_t;

typedef struct { ... } conflux_platform_t;
```

### API

```c
conflux_platform_t *conflux_platform_get(void);
```
获取全局平台实例。

```c
int conflux_platform_init(void);
```
初始化平台（单次有效）。

```c
void conflux_platform_destroy(void);
```
关闭所有设备并清理平台。

```c
int conflux_platform_probe(void);
```
自动探测设备（当前模拟返回 1 个设备）。

```c
int conflux_platform_probe_specific(const conflux_device_desc_t *desc);
```
手动添加设备描述符，用于连接多个 SimX 实例。

```c
int conflux_platform_open_device(int dev_index);
void conflux_platform_close_device(int dev_index);
void conflux_platform_close_all(void);
```
打开/关闭指定设备。打开时会创建设备对象并上线。

```c
int conflux_platform_get_num_devices(void);
conflux_device_t *conflux_platform_get_device(int dev_index);
conflux_device_t *conflux_platform_get_default_device(void);
const conflux_device_desc_t *conflux_platform_get_desc(int dev_index);
```
查询设备。

```c
int conflux_platform_pick_device(void);
```
简单负载均衡：返回第一个未打开的可用设备索引。

```c
void conflux_platform_dump(void);
```
打印平台及所有设备状态。

---

## 9. 日志与性能统计 (`conflux_log.h`)

提供无锁环形日志缓冲和性能计数器。

### 日志 API

```c
void conflux_log_init(conflux_log_level_t level);
```
初始化日志系统，设置最低输出级别。

```c
// 便捷宏
CONFLUX_TRACE(fmt, ...)
CONFLUX_DEBUG(fmt, ...)
CONFLUX_INFO(fmt, ...)
CONFLUX_WARN(fmt, ...)
CONFLUX_ERROR(fmt, ...)
CONFLUX_FATAL(fmt, ...)
```
记录日志。`FATAL` 级别会终止程序。日志同时写入环形缓冲区和 `stderr`（可配置颜色/时间戳）。

```c
void conflux_log_dump_ring(FILE *out);
```
将环形缓冲区内容导出到文件流。

可以通过直接修改 `g_conflux_log` 全局变量来调整时间戳、颜色等。

### 性能统计 API

```c
void conflux_perf_init(void);
```
初始化全局性能统计。

```c
void conflux_perf_record_submit(uint32_t cmd_type);
void conflux_perf_record_copy(size_t bytes);
void conflux_perf_record_alloc(size_t bytes);
void conflux_perf_record_wait(uint64_t wait_ns);
void conflux_perf_record_error(void);
void conflux_perf_record_timeout(void);
```
记录各类事件，用于聚合统计。

```c
void conflux_perf_dump(const conflux_perf_stats_t *stats, char *buf, size_t buf_size);
void conflux_perf_print(void);
```
输出统计报告。

---

*文档版本: 1.0 (适用于 Conflux 运行时库)*