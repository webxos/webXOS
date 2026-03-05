# Makefile for PENCILCLAW v1.0
CXX      = g++
CC       = gcc
CXXFLAGS = -std=c++17 -Wall -Wextra
CFLAGS   = -Wall -Wextra
LDFLAGS  = -lcurl

TARGET = pencilclaw
SOURCES_CPP = pencilclaw.cpp
SOURCES_C = cJSON.c
OBJECTS = pencilclaw.o cJSON.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

pencilclaw.o: pencilclaw.cpp pencil_utils.hpp cJSON.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

cJSON.o: cJSON.c cJSON.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -rf ./pencilclaw/

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
