Introduction

This project using fairseq to train [sentence vectors]. 
and using this vectors to predict next sentence info vector

We make a script to generate [sentence vector], 
and use tensorflow to train the vector to predict the info
in next sentence .
the tensorflow project (https://github.com/youkpan/hadan-tf)

and also try to gen words back ,using the predict vector,
but now facing a problem ,we can't access the hidden layer of dictionary.

fix get hidden layer problem ,and works well :)
anyone find new way improve predict the vector and back to word is welcome. 
my email is youkpan@gmail.com

train file , 170MB of Chinese novel wuxia allbook.txt.xz download： 
[hayoou.com/x31u](http://hayoou.com/x31u)

 -------
 ```
 taining/fconvzh-back is the generate back word model

 using zhgen-line.sh to generate sentence vector

 using python3 train_2.py in another project to train tensorflow

 using generateback.sh to get predict words

 nearly finised  todo: eval tensorflow - > genertaback  online

```

-----install

 first you should follow the fairseq install and use : 
 https://github.com/facebookresearch/fairseq
 then get chinese dataset
 remenber to change ratio 175 in data/prepare-iwslt14.sh  
 for max speed limit the sentence len 75


 our early work :
 
 [predict sentence score](http://f.hayoou.com/blogs/entry/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-caffe-%E6%B5%8B%E8%AF%95%E4%B8%AD)

 [AI-GO (game):](http://f.hayoou.com/blogs/entry/hayoou-AI-GO-%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%9B%B4%E6%A3%8B)

 [OCR Tool (supermarket ticket)](http://f.hayoou.com/blogs/entry/%E5%9F%BA%E4%BA%8E%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9B%BE%E5%83%8F%E6%96%87%E6%9C%AC%E8%AF%86%E5%88%AB%E7%9A%84OCR%E5%B7%A5%E5%85%B7-%E8%AF%86%E5%88%AB%E8%B6%85%E5%B8%82%E5%B0%8F%E7%A5%A8)

 [safe driving](http://f.hayoou.com/blogs/entry/%E5%93%88%E5%8F%8B%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%AE%89%E5%85%A8%E8%BE%85%E5%8A%A9%E9%A9%BE%E9%A9%B6%E5%BA%94%E7%94%A8)
