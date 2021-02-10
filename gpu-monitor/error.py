ERRORLOG='./logs/error.log'
# should never occurr...
def errorLog(sentence):
    f=open(ERRORLOG, 'a+')
    f.write("ERROR: "+sentence)
    f.write("\n")
    f.flush()
    f.close()
