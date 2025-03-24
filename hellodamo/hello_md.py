from modelscope.pipelines import pipeline
import numpy as np
from packaging import version
import addict

input = '昨天起，上海地铁3号线长江南路站、殷高西路站、江湾镇站三站进一步限流。体验发现，高峰时段排队5分钟能进站；不少乘客选择提前起床，“现在提前10到20分钟起床，即便限流也不会影响上班”。被限流的XDJMS，你们提前多久？新民网'
text_summary = pipeline('text-generation', model='damo/nlp_palm2.0_text-generation_chinese-base')
out=text_summary(input)
print(out)

