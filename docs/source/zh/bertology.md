<!--版权2020年HuggingFace团队保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则按“按原样”分发的软件，无论是明示还是暗示的，都没有任何担保或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️ 请注意，本文件虽然使用Markdown编写，但包含了特定的语法，适用于我们的doc-builder（类似于MDX），可能无法在您的Markdown查看器中正常渲染。

-->

# 基于BERT进行的相关研究（BERTology）

当前，一个新兴的研究领域正致力于探索大规模 transformer 模型（如BERT）的内部工作机制，一些人称之为“BERTology”。以下是这个领域的一些典型示例：


- BERT Rediscovers the Classical NLP Pipeline by Ian Tenney, Dipanjan Das, Ellie Pavlick:
  https://arxiv.org/abs/1905.05950
- Are Sixteen Heads Really Better than One? by Paul Michel, Omer Levy, Graham Neubig: https://arxiv.org/abs/1905.10650
- What Does BERT Look At? An Analysis of BERT's Attention by Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D.
  Manning: https://arxiv.org/abs/1906.04341
- CAT-probing: A Metric-based Approach to Interpret How Pre-trained Models for Programming Language Attend Code Structure: https://arxiv.org/abs/2210.04633


为了助力这一新兴领域的发展，我们在BERT/GPT/GPT-2模型中增加了一些附加功能，方便人们访问其内部表示，这些功能主要借鉴了Paul Michel的杰出工作(https://arxiv.org/abs/1905.10650)：


- 访问BERT/GPT/GPT-2的所有隐藏状态，
- 访问BERT/GPT/GPT-2每个注意力头的所有注意力权重，
- 检索注意力头的输出值和梯度，以便计算头的重要性得分并对头进行剪枝，详情可见论文：https://arxiv.org/abs/1905.10650。

为了帮助您理解和使用这些功能，我们添加了一个具体的示例脚本：[bertology.py](https://github.com/huggingface/transformers/tree/main/examples/research_projects/bertology/run_bertology.py)，该脚本可以对一个在 GLUE 数据集上预训练的模型进行信息提取与剪枝。