
# 当前目录makefile文件
CXX = g++
CC = gcc

# 编译选项
CXXFLAGS = -g -Wall -std=c++11
CFLAGS = -g -Wall

# 目标文件
TARGET = cnn

# 源文件
SRCS = $(wildcard *.c)
OBJS = $(patsubst %.c, %.o, $(SRCS))

# 生成目标文件
$(TARGET): $(OBJS)
	$(CC) $(CXXFLAGS) -o $@ $^


# 生成目标文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<
%.o: %.c
	$(CC) $(CFLAGS) -c $<
# 清理
clean:
	rm -f $(OBJS) $(TARGET)

