CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=gnu99
LDFLAGS = -lcurl -lm

TARGET = shadowclaw
OBJS = shadowclaw.o cJSON.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

strip: $(TARGET)
	strip $(TARGET)

.PHONY: all clean strip
