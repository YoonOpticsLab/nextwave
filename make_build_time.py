import datetime
import os

USE_GIT_LOG=True

if USE_GIT_LOG:
    os.system("git log -n1 > gitlog.txt")
    fil=open("gitlog.txt","rt")
    lins=fil.readlines()

    fil=open("nextwave_build.py","wt")
    # LAst git commit ID, date, message
    fil.writelines("build_message='Version info:\\n%s\\n%s\\n%s'"%(lins[0].strip(),lins[2].strip(),lins[4].strip() ) ) 
    fil.close()
    
else:
    extra_info="STANDALONE OFFLINE"
    extra_info2="Zero saturated. Editable defaults.py"

    fil=open("nextwave_build.py","wt")
    str_now = datetime.datetime.now()
    fil.writelines("build_message='%s\\n%s\\n(%s)'"%(str(str_now),extra_info,extra_info2) )
    fil.close()

