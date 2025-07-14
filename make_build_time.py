import datetime

extra_info="STANDALONE OFFLINE"
extra_info2="Start expand smaller pupil."


fil=open("nextwave_build.py","wt")
str_now = datetime.datetime.now()
fil.writelines("build_message='%s\\n%s\\n(%s)'"%(str(str_now),extra_info,extra_info2) )
fil.close()