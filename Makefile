ROOT   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD  := $(ROOT)/build
CC     := clang
CFLAGS := -std=gnu17 -O2 -g -Wall -Wextra

all: $(BUILD)/wcu-translate $(BUILD)/libwcu.a

$(BUILD)/wcu-translate: src/translate.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD)/runtime.o: src/runtime.c include/wcu.h | $(BUILD)
	$(CC) $(CFLAGS) -Iinclude -DWCU_DEFAULT_VIEWER='"$(ROOT)/viewer/viewer.html"' -c -o $@ $<

$(BUILD)/libwcu.a: $(BUILD)/runtime.o
	rm -f $@ && ar rcs $@ $^

$(BUILD):
	mkdir -p $@

site: all
	@bash docs/build-demos.sh

test: all
	@bash tests/run.sh

clean:
	rm -rf $(BUILD)

.PHONY: all test clean site
