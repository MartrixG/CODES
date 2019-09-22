#include <cstdio>
struct Node{ 
   int key;
   int priority;
   Node* left;
   Node* right;
};
Node* root;
void rotate_left(Node* &node)
{
	Node* x = node->right;
	node->right = x->left;
	x->left =node;
	node = x;
}
void rotate_right(Node* &node)
{
	Node* x = node->left;
	node->left = x->right;
	x->right = node;
	node = x;
}

void treap_insert(Node* &root, int key, int priority)
{
	if (root == NULL)
	{
		root = (Node*)new Node;
		root->left = NULL;
		root->right = NULL;
		root->priority = priority;
		root->key = key;
	}
	else if (key <root->key)
	{
		treap_insert(root->left, key, priority);
		if (root->left->priority < root->priority)
			rotate_right(root);
	}
	else
	{
		treap_insert(root->right, key, priority);
		if (root->right->priority < root->priority)
			rotate_left(root);
	}
}
void treap_delete(Node* &root, int key)
{
	if (root != NULL)
	{
		if (key < root->key)
			treap_delete(root->left, key);
		else if (key > root->key)
			treap_delete(root->right, key);
		else
		{
			if (root->left == NULL)//左孩子为空
				root = root->right;
			else if (root->right == NULL) //右孩子为空
				root = root->left;
			else//左右孩子均不为空
			{
				if (root->left->priority < root->right->priority)//先旋转，然后再删除
				{
					rotate_right(root);
					treap_delete(root->right, key);
				}
				else
				{
					rotate_left(root);
					treap_delete(root->left,key);
				}
			}
		}
	}
}
//中序遍历
void in_order_traverse(Node* root)
{
	if (root!= NULL)
	{
		in_order_traverse(root->left);
		printf("%d\t", root->key);
		in_order_traverse(root->right);
	}
}
//计算树的高度
int depth(Node* node)
{
    if(node == NULL)
        return -1;
    int l = depth(node->left);
    int r = depth(node->right);
    return (l < r)?(r+1):(l+1);
}
int rand()
{
    static int seed = 703;
    return seed = int(seed*48271LL%(~0u>>1));
}
int main()
{
	printf("----------------------创建Treap树堆-----------------------\n");
	printf("顺序插入0至9十个数，键值与优先级如下\n");
	for (int i = 0; i < 10; i++)
	{
		int pri=rand();
		printf("key:%d priority:%d\n",i,pri);
		treap_insert(root,i,pri);
	}
	printf("\n插入完毕，中序遍历Treap所得结果为:\n");
	in_order_traverse(root);
	printf("\nTreap高度：%d\n", depth(root));
	printf("----------------------删除结点-----------------------\n");
	printf("请输入要删除的结点键值\n");
	int rmKey;
	scanf("%d",&rmKey);
	treap_delete(root, rmKey);
	printf("\n删除完毕，中序遍历Treap所得结果为:\n");
	in_order_traverse(root);
	printf("\nTreap高度：%d\n", depth(root));
	getchar();
	getchar();
	return 0;
}