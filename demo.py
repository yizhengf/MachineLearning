import re
pattern1="cat"
pattern2="bird"
string="dogs run to cat"
print(re.search(pattern1,string))
print(re.search(pattern2,string))