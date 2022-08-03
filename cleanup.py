import os
import time
import shutil
while True:
    numdays = 86400*7
    now = time.time()
    directory=os.path.join("/home/dickinsonq/mimal/")
    print(directory)
    for r,d,f in os.walk(directory):
        print(r)
        for dir in d:
            timestamp = os.path.getmtime(os.path.join(r,dir))
            #if now-numdays > timestamp or len(os.listdir(os.path.join(r,dir))) == 0:
            if len(os.listdir(os.path.join(r,dir))) == 0:
                try:
                    print("removing ",os.path.join(r,dir))
                    shutil.rmtree(os.path.join(r,dir)) 
                except (Exception,e):
                    print (e)
                    pass
                else: 
                    print ("some message for success")

    time.sleep(86400)