# Protein_Mel_ML

这是一个蛋白质转换的项目，还没有完善

我们将FASTA转换成WAV音频，然后识别WAV音频生成MEL频谱图片

下面是转换的代码文件

[ftow220.py](https://github.com/wyqmath/protein_mel_ml/blob/master/ftow220.py)

然后通过随机森林、XGboost、SVM、MLP四个模型的集合，实现了对蛋白质的分类

准确率可达90%，交叉验证分数0.95，处理2000个样本仅需150s，效率极高

下面是分类的代码文件

[proteinml.py](https://github.com/wyqmath/protein_mel_ml/blob/master/proteinml.py)

数据集来源于：
https://www.ncbi.nlm.nih.gov/protein

其余数据集的FASTA文件也是可用的，例如：

https://www.rcsb.org/