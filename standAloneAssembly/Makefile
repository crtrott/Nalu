KOKKOS_PATH = ../packages/Trilinos/packages/kokkos
SRC = $(wildcard *.C)
FSRC = $(wildcard *.F)

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = nvcc_wrapper
CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS = 
EXE = $(SRC:.cpp=.cuda)
KOKKOS_DEVICES = "Cuda,OpenMP"
KOKKOS_ARCH = "SNB,Kepler35"
else
CXX = g++
CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =  
EXE = standAloneNalu
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "SNB"
endif
F90 = gfortran
DEPFLAGS = -M

OBJ = $(SRC:.C=.o)
OBJ += $(FSRC:.F=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.C $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

%.o:%.F 
	$(F90) -c $<
