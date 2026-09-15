ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD  := $(ROOT)/build
CC     := clang
CFLAGS := -std=gnu17 -O2 -g -Wall -Wextra

all: $(BUILD)/cuemu-translate $(BUILD)/libcuemu.a

$(BUILD)/cuemu-translate: src/translate.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD)/runtime.o: src/runtime.c include/cuemu.h | $(BUILD)
	$(CC) $(CFLAGS) -Iinclude -DCUEMU_DEFAULT_VIEWER='"$(ROOT)/viewer/viewer.html"' -c -o $@ $<

$(BUILD)/libcuemu.a: $(BUILD)/runtime.o
	rm -f $@ && ar rcs $@ $^

$(BUILD):
	mkdir -p $@

test: all
	@bash tests/run.sh

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
