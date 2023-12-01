import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages

def gen_pdf_summary(pdfFilename):
    print(f"Generate pdf at : {pdfFilename}")
    # PdfPages is a wrapper around pdf 
    # file so there is no clash and
    # create files with no error.
    p = PdfPages(pdfFilename)
      
    # get_fignums Return list of existing
    # figure numbers
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
      
    # iterating over the numbers in list
    for fig in figs: 
        
        # and saving the files
        fig.savefig(p, format='pdf') 
          
    # close the object
    p.close()  