#include <stdio.h>
#include <string.h>
#include <ctype.h>
#define MAXCHARS 999

typedef char word_t[MAXCHARS + 1];
int GetWord(word_t R, int limit);
int GetLine(char L[], int maxchar);
int mygetchar();
//oneLinelim（字符数目限制）spaces（空格数）nowPos当前这一行的字符数目
int oneLinelim = 50, spaces = 4, nowPos = 0, k;

int main(int argc, char *argv[])
{
    //freopen("1.txt", "r", stdin);
    freopen("11.txt", "w", stdout);
    word_t one_line, without_command_line[MAXCHARS];
    int i = 0;
    printf("    ");
    nowPos = 4;
    while (GetLine(one_line, MAXCHARS) != EOF)
    {
        if (one_line[0] == '.')
        {
            int flag;
            //处理同时出现.p .w 12:
            if (one_line[1] == 'p' && one_line[2] == ' ' \
                && one_line[3] == '.' && one_line[4] == 'p')
            {
                flag = 7;
                char num1 = one_line[6], num2 = one_line[7];
                spaces = num1 - '0';
                if (num2 >= '0' && num2 <= '9')
                {
                    spaces *= 10;
                    spaces += num2 - '0';
                    flag++;
                }
                printf("\n\n");
                for (k = 0; k < spaces; k++)
                {
                    printf(" ");
                }
                nowPos = spaces;
            }
            //处理.b
            if (one_line[1] == 'b')
            {
                printf("\n");
                for (k = 0; k < spaces; k++)
                {
                    printf(" ");
                }
                flag = 2;
                nowPos = spaces;
            }
            //处理.p
            if (one_line[1] == 'p')
            {
                printf("\n\n");
                for (k = 0; k < spaces; k++)
                {
                    printf(" ");
                }
                flag = 2;
                nowPos = spaces;
            }
            //处理.w
            if (one_line[1] == 'w')
            {
                printf("\n\n");
                flag=4;
                for (k = 0; k < spaces; k++)
                {
                    printf(" ");
                }
                char num1 = one_line[3], num2 = one_line[4];
                oneLinelim = num1 - '0';
                if (num2 >= '0' && num2 <= '9')
                {
                    oneLinelim *= 10;
                    oneLinelim += num2 - '0';
                    flag++;
                }
                nowPos = spaces;
            }
            //处理.l
            if (one_line[1] == 'l')
            {
                printf("\n\n");
                flag = 4;
                char num1 = one_line[3], num2 = one_line[4];
                spaces = num1 - '0';
                if (num2 >= '0' && num2 <= '9')
                {
                    spaces *= 10;
                    spaces += num2 - '0';
                    flag++;
                }
                for (k = 0; k < spaces; k++)
                {
                    printf(" ");
                }
                nowPos = spaces;
            }
            //处理.c .h
            if (one_line[1] == 'c' || one_line[1] == 'h')
            {
                continue;
            }
            int len = strlen(one_line);
            //如果仅有控制符直接跳过之后的
            if(flag == len)
            {
                continue;
            }
            //删去开头的控制符
            for (k = flag; k < len; k++)
            {
                one_line[k - flag] = one_line[k];
            }
        }
        //如果当前这一行为空直接跳过
        if (strlen(one_line) == 0)
            continue;
        strcpy(without_command_line[i++], one_line);
        GetWord(without_command_line[i - 1], strlen(without_command_line[i - 1]));
    }
    return 0;
}

int GetWord(word_t R, int limit)
{
    //补空格输出最后一个单词
    R[limit++] = ' ';
    int now = 0;
    int lenWord = 0;
    word_t tempWord;
    while (now < limit)
    {
        if (R[now] != ' ' && R[now] != '\t')
        {
            tempWord[lenWord++] = R[now];
            tempWord[lenWord] = '\0';
        }
        if (R[now] == ' ' || R[now] == '\t')
        {
            //如果当前的词长度不是0并且读到了空格
            if (lenWord != 0)
            {
                //刚好相同输出没有空格
                if (lenWord + nowPos == oneLinelim)
                {
                    printf("%s", tempWord);
                    nowPos += lenWord;
                }
                //小于输出带空格
                else if (lenWord + nowPos < oneLinelim)
                {
                    printf("%s ", tempWord);
                    nowPos += lenWord + 1;
                }
                //长度不够
                else if (lenWord + nowPos > oneLinelim)
                {
                    //换行输出空格
                    printf("\n");
                    for (k = 0; k < spaces; k++)
                    {
                        printf(" ");
                    }
                    nowPos = spaces;
                    //如果依然超过，则这个词过长，直接输出、换行、加空格
                    if (lenWord + nowPos > oneLinelim)
                    {
                        printf("%s\n", tempWord);
                        for (k = 0; k < spaces; k++)
                        {
                            printf(" ");
                        }
                        nowPos = spaces;
                    }
                    else
                    {
                        printf("%s ", tempWord);
                        nowPos += lenWord;
                    }
                }
            }
            lenWord = 0;
        }
        now++;
    }
    return 0;
}

int GetLine(char L[], int maxchar)
{
    int c, charlen = 0;
    while ((c = mygetchar()) != EOF && charlen < maxchar)
    {
        //出现换行直接跳出，换行由GetWord执行。
        if (c == '\n')
        {
            break;
        }          
        L[charlen] = c;
        charlen++;
      
    }
    if (c == EOF && charlen == 0)
    {
        return EOF;
    }
    L[charlen] = '\0';
    return 0;
}

int mygetchar()
{
    char c;
    while ((c = getchar()) == '\r')
    {
    }
    return c;
}