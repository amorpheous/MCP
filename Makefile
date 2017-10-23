# C-Makefile for Seth
CC     = mpiCC #/opt/lam/gnu/bin/mpiCC      #/usr/local/mpich/bin/mpiCC
CFLAGS = -O3 -I/opt/mpich/gnu/include -lm
#-I/opt/scali/include -lm
LIBS =  -L/opt/mpich/gnu/lib -lpthread

# Define the application object files and target name
#   APPOBJ = list of required object files
#   APP    = name of target executable

APPOBJ = MCP.cpp
APP = MCP.exe

$(APP): $(APPOBJ)
	$(CC) $(CFLAGS) -o $(APP) $(APPOBJ) $(LIBS)

clean:
	/bin/rm -f $(APP) $(APPOBJ) 
