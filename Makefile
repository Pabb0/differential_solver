CC = g++
CFLAGS = -std=c++17 -O3 -DNDEBUG

simulation1.o: simulation1.cpp ivp_pandemic.hpp ivp_solvers.hpp
	$(CC) -c $(CFLAGS) simulation1.cpp

simulation1: simulation1.o
	$(CC) -o simulation1 simulation1.o

simulation2.o: simulation2.cpp ivp_solvers.hpp
	$(CC) -c $(CFLAGS) simulation2.cpp

simulation2: simulation2.o
	$(CC) -o simulation2 simulation2.o


estimation1.o: estimation1.cpp bfgs.hpp gradient.hpp linesearch.hpp lse.hpp ivp_pandemic.hpp ivp_solvers.hpp
	$(CC) -c $(CFLAGS) estimation1.cpp

estimation1: estimation1.o
	$(CC) -o estimation1 estimation1.o

estimation2.o: estimation2.cpp bfgs.hpp gradient.hpp linesearch.hpp lse.hpp ivp_pandemic.hpp ivp_solvers.hpp
	$(CC) -c $(CFLAGS) estimation2.cpp

estimation2: estimation2.o
	$(CC) -o estimation2 estimation2.o