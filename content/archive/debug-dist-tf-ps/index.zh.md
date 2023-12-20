---
layout: post
title: "Debug Tensorflow PS的数据传输"
date: "2018-09-27 11:00:00 +0800"
---

在分布式训练中，有时会碰到PS传输数据量很大的情况。这时候，可以在代码中加log来帮助找出哪个tensor消耗比较大。

在grpc_remote_worker.cc中添加如下代码:

```c++
void RecvTensorAsync(CallOptions* call_opts, const RecvTensorRequest* request,
                       TensorResponse* response, StatusCallback done) override {
    VLOG(1) << "RecvTensorAsync req: " << request->DebugString();
    int64 start_usec = Env::Default()->NowMicros();
    // Type-specialized logging for this method.
    bool logging_active = logger_->LoggingActive() || VLOG_IS_ON(2) || true;
    StatusCallback wrapper_done;
    const StatusCallback* cb_to_use;
    if (!logging_active) {
      cb_to_use = &done;  // No additional work to do, so just use done directly
    } else {
      wrapper_done = [this, request, response, done, start_usec](Status s) {
        int64 bytes = response->tensor().TotalBytes();
        const string& key = request->rendezvous_key();
        std::vector<string> key_parts = str_util::Split(key, ';');
        LOG(INFO) << "recv tensor name: " << key_parts[3] << " src: " << key_parts[0] << " dest: " << key_parts[2] << " bytes: " << bytes;
```


就可以输出日志:

recv tensor name: xx src: /job:ps/replica:0/task:0/device:CPU:0 dest: /job:worker/replica:0/task:0/device:CPU:0 bytes: 20889600

这样从src传输到dest, tensor名称，大小都可以看到了。

