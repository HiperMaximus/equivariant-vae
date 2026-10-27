#define _GNU_SOURCE
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>

/* Synthetic semantic check only: no Kaggle/NFS performance is measured. */
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    size_t page = (size_t)sysconf(_SC_PAGESIZE), total = 5 * page;
    int a = open(argv[1], O_RDONLY), b = open(argv[2], O_RDONLY);
    if (a < 0 || b < 0) { perror("open"); return 2; }
    void *arena = mmap(NULL, total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (arena == MAP_FAILED) { perror("reserve"); return 2; }
    struct { int fd; size_t source, destination, pages; } pieces[] = {
        {a, 3, 0, 2}, {b, 1, 2, 2}, {a, 0, 4, 1}
    };
    for (size_t i = 0; i < 3; i++) {
        /* MAP_FIXED replaces only our own reserved, page-aligned subrange. */
        void *mapped = mmap((char *)arena + pieces[i].destination * page,
                            pieces[i].pages * page, PROT_READ, MAP_PRIVATE | MAP_FIXED,
                            pieces[i].fd, (off_t)(pieces[i].source * page));
        if (mapped == MAP_FAILED) { perror("map piece"); return 2; }
    }
    close(a); close(b);
    if (madvise(arena, total, MADV_SEQUENTIAL) != 0) { perror("madvise"); return 2; }
    const unsigned char expected[] = {3, 4, 101, 102, 0};
    const volatile unsigned char *view = arena;
    for (size_t i = 0; i < total; i++)
        if (view[i] != expected[i / page]) return 3;
    printf("{\"virtual_contiguous\":true,\"synthetic_bytes_equal\":true,"
           "\"mapped_payload_bytes\":%zu,\"mapped_regions\":3,"
           "\"source_open_read_only\":true,\"payload_copy_before_access\":false,"
           "\"performance_measured\":false}\n", total);
    if (munmap(arena, total) != 0) { perror("unmap"); return 2; }
    return 0;
}
