> 保守组合:均严格按照官方要求的支持的稳定版本.稳定性可能高些但性能可能低些.
>         版本组合:cuda12.1/pytorch2.1.0/python3.10/auto-gptq0.5.1/vllm0.5.4
> 
> pip install auto-gptq==0.5.1 #一定要指定安装版本
> pip install vllm==0.5.4  # 注意:vllm安装的同时,CUDA 12.1和pytorch将被自动再安装一次,无须理会!!!
————————————————
 
原文链接：https://blog.csdn.net/weixin_42745482/article/details/145277526