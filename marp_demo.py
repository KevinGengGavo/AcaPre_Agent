from lib import slides_suggestion

pdfpath = "./output/slides.pdf"

res = slides_suggestion.trans2marp(pdfpath)

print(res)