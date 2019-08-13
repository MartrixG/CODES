
#include<bits/stdc++.h>

#define N 1000
using namespace std;
int Q;

struct BST
{
	BST *Left,*Right;
	int num;
}*rt=NULL;

void Insert(BST *&x,int v) 
{
	if(!x)//若当前节点为空 
	{
		x=new BST;//新建一个节点 
		x->Left=x->Right=NULL,x->num=v;//将这个新节点设定为插入元素 
		return;
	}
	if(v==x->num) return;//若已插入过，则退出函数 
	if(v<x->num) Insert(x->Left,v);//若插入元素小于当前节点元素，则继续对当前节点的左子树进行操作
	else Insert(x->Right,v);//反之，继续对当前节点的右子树进行操作
}

int Query(BST *x,int v)
{
	if(!x) return 0;//若当前节点为空，则返回0 
	if(v==x->num) return 1;//若查询元素与当前节点相等，则返回1 
	return v<x->num?Query(x->Left,v):Query(x->Right,v);
}

void Delete(BST *&x,int v)
{
	if(!x) return;//若当前节点为空，则退出函数 
	if(v==x->num) {x->Left->Right=x->Right,x=x->Left;return;}//删除当前节点 
	if(v<x->num) Delete(x->Left,v);
	else Delete(x->Right,v);
}

int main()
{
	scanf("%d",&Q);
	for(int i=1;i<=Q;i++) {
		int x,y;
		scanf("%d%d",&x,&y);
		if(x==1) Insert(rt,y);
		if(x==2) printf("%d\n",Query(rt,y));
		if(x==3) Delete(rt,y);
	}
	return 0;
}
