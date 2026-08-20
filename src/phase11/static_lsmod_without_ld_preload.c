#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/*
 * libmigraphx_gpu.so is globally preloaded by the matched vendor image.  Its
 * process initializer invokes `lsmod`; inheriting LD_PRELOAD into the dynamic
 * kmod binary recursively repeats that initializer.  This static shim ignores
 * LD_PRELOAD itself, removes it for the child, and then executes the real kmod
 * with an argv[0] whose basename remains `lsmod`.
 */
int main(int argc, char **argv) {
    (void)argc;
    if (unsetenv("LD_PRELOAD") != 0) {
        perror("unsetenv(LD_PRELOAD)");
        return 125;
    }
    execv("/bin/kmod", argv);
    fprintf(stderr, "execv(/bin/kmod) failed: errno=%d\n", errno);
    return 126;
}
