#ifndef CONFLUX_ERROR_H
#define CONFLUX_ERROR_H

typedef enum {
    CONFLUX_SUCCESS = 0,
    
    /* 通用错误 (-1 到 -9) */
    CONFLUX_ERR_INVALID  = -1,   /* 参数无效（NULL、越界等） */
    CONFLUX_ERR_NOMEM    = -2,   /* 主机内存不够（malloc 失败） */
    CONFLUX_ERR_TIMEOUT  = -3,   /* 操作超时 */
    CONFLUX_ERR_BUSY     = -4,   /* 设备正忙 */
    
    /* 内存模块 (-10 到 -19) */
    CONFLUX_ERR_MEM_OUT_OF_DEVICE = -10,  /* 设备显存不够 */
    CONFLUX_ERR_MEM_FRAGMENTED    = -11,  /* 碎片导致分配失败 */
    CONFLUX_ERR_MEM_INVALID_ADDR  = -12,  /* 设备地址无效 */
    CONFLUX_ERR_MEM_OVERLAP       = -13,  /* 地址重叠 */
    
    /* 事件模块 (-20 到 -29) */
    CONFLUX_ERR_EVENT_INVALID_STATE = -20, /* 事件状态转换不合法 */
    CONFLUX_ERR_EVENT_NULL           = -21, /* 事件指针为空 */
    
    /* 设备模块 (-30 到 -39) */
    CONFLUX_ERR_DEVICE_NOT_READY   = -30, /* 设备未初始化 */
    CONFLUX_ERR_DEVICE_FAULT       = -31, /* 设备硬件错误 */
    CONFLUX_ERR_DEVICE_RESET_FAIL  = -32, /* 复位失败 */
    
    /* 命令相关 (-40 到 -49) */
    CONFLUX_ERR_CMD_INVALID_OPCODE = -40, /* 不支持的命令码 */
    CONFLUX_ERR_CMD_QUEUE_FULL     = -41, /* 命令队列满 */
    CONFLUX_ERR_CMD_ABORTED        = -42, /* 命令被中止 */
    
} conflux_error_t;

/* 把错误码转成可读字符串 */
static inline const char *conflux_strerror(int err) {
    switch (err) {
        case CONFLUX_SUCCESS:                return "Success";
        case CONFLUX_ERR_INVALID:            return "Invalid argument";
        case CONFLUX_ERR_NOMEM:              return "Host out of memory";
        case CONFLUX_ERR_TIMEOUT:            return "Timeout";
        case CONFLUX_ERR_BUSY:               return "Device busy";
        case CONFLUX_ERR_MEM_OUT_OF_DEVICE:  return "Device out of memory";
        case CONFLUX_ERR_MEM_FRAGMENTED:     return "Memory fragmented";
        case CONFLUX_ERR_MEM_INVALID_ADDR:   return "Invalid device address";
        case CONFLUX_ERR_MEM_OVERLAP:        return "Memory overlap";
        case CONFLUX_ERR_EVENT_INVALID_STATE:return "Invalid event state";
        case CONFLUX_ERR_EVENT_NULL:         return "Null event pointer";
        case CONFLUX_ERR_DEVICE_NOT_READY:   return "Device not ready";
        case CONFLUX_ERR_DEVICE_FAULT:       return "Device hardware fault";
        case CONFLUX_ERR_DEVICE_RESET_FAIL:  return "Device reset failed";
        case CONFLUX_ERR_CMD_INVALID_OPCODE: return "Invalid command opcode";
        case CONFLUX_ERR_CMD_QUEUE_FULL:     return "Command queue full";
        case CONFLUX_ERR_CMD_ABORTED:        return "Command aborted";
        default:                            return "Unknown error";
    }
}

#endif