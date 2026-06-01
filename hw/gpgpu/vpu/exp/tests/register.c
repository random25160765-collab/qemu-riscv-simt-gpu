/*
 * tests/register.c — Central test registration (call once)
 */
extern void func_tests_register(void);
extern void bench_tests_register(void);

void tests_register(void)
{
    static int done = 0;
    if (done) return;
    done = 1;
    func_tests_register();
    bench_tests_register();
}
