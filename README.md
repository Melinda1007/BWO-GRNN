# BWO-GRNN
part 1
pyBWO中的BWO属于对GRNN优化参数的部分
此部分已经调试通，但是目前的数据集只限于单个输入和输出，后续需要改进

11.15发现问题
只进行一次循环，第二次循环出现问题：list index out of range 

part 2
下一步的输入：肌电特征值（iEMG、RMS）和角度信息
做实验和数据集构建

更新代码
已经可以实现程序流程，即BWOA的版本
注：文件夹中的文件主要是在编码过程中的参考的代码，以及创建的过程，仅仅作为个人的记录
