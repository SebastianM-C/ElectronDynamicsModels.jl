using PDFIO

# Open the PDF document
doc = pdDocOpen("poster.pdf")

# Get the number of pages
num_pages = pdDocGetPageCount(doc)

println("Number of pages in poster.pdf: $num_pages")

# Close the document
pdDocClose(doc)