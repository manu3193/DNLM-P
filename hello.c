#include <stdio.h>
#include "ipps.h"
const char* test_str = "Hello Intel(R) IPP on Intel(R) Xeon Phi(TM)!";
int main()
{
    int len;
    Ipp8u* ipp_buf;

    len = strlen((void*)test_str);
    ipp_buf = ippsMalloc_8u(len+1);
    ippsCopy_8u((const Ipp8u*)test_str, ipp_buf, len);
    ipp_buf[ len+1 ] = 0;
    printf("Test string: %s\n", ipp_buf);
    ippsFree(ipp_buf);
}
