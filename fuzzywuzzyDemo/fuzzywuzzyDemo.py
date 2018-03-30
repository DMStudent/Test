# --coding:utf-8 --
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

text1 = u"百度是一家高科技公司"
text2 = u"发丝发生发生的v发生的!"

print (float(fuzz.ratio(text1, text2))/100)
print fuzz.partial_ratio(text1, text2)
print fuzz.token_sort_ratio(text1, text2, force_ascii=True)
print fuzz.token_set_ratio(text1, text2, force_ascii=False)
print fuzz.UWRatio(text1, text2)