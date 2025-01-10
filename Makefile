

CUDA_PATH ?= /usr/local/cuda
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin g++

NVCCFLAGS   := -m64
CCFLAGS     :=
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += --threads 0 --std=c++17

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))


INCLUDES += -I./Common -I./Common/UtilNPP
LIBRARIES += -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage




all: build

build: edgeDetector

%.o : src/%.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

edgeDetector: edgeDetector.o
	$(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)

run: build
	./edgeDetector inputs outputs

clean:
	rm -f edgeDetector edgeDetector.o

