import re
import locale

content = "ab* hilary_ 48838dw @we DDFFD3*"

print re.split("([`|~!@#\$%\^&\*\(\)\-_=\+\[\{\]\};:'\",<\.>/\?\s])",
               content)
print re.split("(\w+)|", content)

print locale.getdefaultlocale()

numbers = [x for x in range(10)]
print numbers




