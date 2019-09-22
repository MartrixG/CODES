#include <stdio.h>
#include <string.h>
int main()
{
    freopen("11.txt", "w", stdout);
    char s[1000];
    int now = 0, flag = 1;
    while (scanf("%s", s) != EOF)
    {
        if (flag)
        {
            printf("    ");
            now += 4;
            flag = 0;
        }
        int len = strlen(s);
        if (now + len > 50)
        {
            now = 4;
            printf("\n    ");
        }
        now += len + 1;
        printf("%s ", s);
    }
}