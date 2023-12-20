---

layout: post

title: "二级指针在链表操作中的应用"

date: 2013-07-20 19:44

comments: true

categories: C

---



故事起源于Linus大神在回答水友提问的时候有这么一段: <http://meta.slashdot.org/story/12/10/11/0030249/linus-torvalds-answers-your-questions>



At the opposite end of the spectrum, I actually wish more people understood the really core low-level kind of coding. Not big, complex stuff like the lockless name lookup, but simply good use of pointers-to-pointers etc. For example, I've seen too many people who delete a singly-linked list entry by keeping track of the "prev" entry, and then to delete the entry, doing something like

 

{% highlight c %}

if (prev)

    prev->next = entry->next;

else

    list_head = entry->next;

{% endhighlight %}



and whenever I see code like that, I just go "This person doesn't understand pointers". And it's sadly quite common.

 

People who understand pointers just use a "pointer to the entry pointer", and initialize that with the address of the list_head. And then as they traverse the list, they can remove the entry without using any conditionals, by just doing a "*pp = entry->next".



大意是说，当从一个单向链表中删除一个元素时，很多人会使用一个prev指针来记录被删除的那个节点的前一个元素的位置，然后使用`prev->next = entry->next`的方式去删除。这样的人简直弱爆了，根本就不懂得指针。



可能这样说还不是很清楚，让我们来看个比较完整的例子吧：



{% highlight c %}

Node* find_and_delete(Node *head, int target)

{

    Node *prev = NULL;

    Node *entry = head;

    while (entry != NULL) {

        if (entry->val == target) {

            if (prev) {

                prev->next = entry->next;

            } else {

                head = entry->next;

            }   

            break;

        }   

        prev = entry;

        entry = entry->next;

    }   

    return head;

}

{% endhighlight %}



函数`find_and_delete`完成了这么一个功能：遍历一个单向链表，如果找到和给出的target相等的值，就从链表中删除这个节点。上述方法是一般c语言教科书（例如谭浩强之类）里的标准做法。但是这种方法遭到了Linus大神的强烈鄙视和唾弃，认为这是不懂指针的人的做法。Linus提到了一个使用二级指针的方法`*pp = entry->next`可以让我们不需要判断就可以删除，这是怎么做到的呢？



{% highlight c %}

void find_and_delete2(Node **head, int target)

{

    while (*head != NULL) {

        Node *entry = *head;

        if (entry->val == target) {

            *head = entry->next;

            break;

        }   

        head = &(entry->next);

    }   

}

{% endhighlight %}



在`find_and_delete2`函数中，巧妙的使用了一个二级指针，从而直接修改了当前entry指针的指向，代码精简了很多。不过这个代码并非很容易正确实现（大家可以自己试试，能不能一遍写正确）。(特别的，想想`head = &(entry->next)，`和`*head = entry->next`有什么区别？ )



不得不说玩kernel的大神对指针的理解确实比我辈强的多，学一招还挺有用的。学了这招以后去leetcode上切了一题，也用了二级指针:-)



<https://github.com/codescv/leetcode/blob/master/AddTwoNumbersNoNew.cpp>



最后，上个完整的测试代码例子，供懒得敲代码的人玩耍：



{% highlight c %}

#include <stdio.h>

#include <stdlib.h>

#include <string.h>



struct NodeStruct {

    int val;

    struct NodeStruct *next;

};



typedef struct NodeStruct Node;



Node* find_and_delete(Node *head, int target)

{

    Node *prev = NULL;

    Node *entry = head;

    while (entry != NULL) {

        if (entry->val == target) {

            if (prev) {

                prev->next = entry->next;

            } else {

                head = entry->next;

            }

            break;

        }

        prev = entry;

        entry = entry->next;

    }

    return head;

}



void find_and_delete2(Node **head, int target)

{

    while (*head != NULL) {

        Node *entry = *head;

        if (entry->val == target) {

            *head = entry->next;

            break;

        }

        head = &(entry->next);

    }

}



Node *make_list(int a[], int n)

{

    Node *result = NULL;

    Node *runner;

    int i;

    if (n == 0) {

        return NULL;

    }

    result = (Node *)malloc(sizeof(Node));

    result->val = a[0];

    runner = result;

    for (i = 1; i < n; i++) {

        runner->next = (Node *)malloc(sizeof(Node));

        runner->next->val = a[i];

        runner = runner->next;

    }

    return result;

}



void print_list(Node *l)

{

    while (l) {

        printf("%d", l->val);

        if (l->next) {

            printf("->");

        } else {

            printf("\n");

        }

        l = l->next;

    }

}



int main(int argc, const char *argv[])

{

    int a[] = {1,2,3,4,5};

    Node *l = make_list(a, 5);

    print_list(l);

    l = find_and_delete(l, 3);

    print_list(l);

    l = find_and_delete(l, 1);

    print_list(l);

    Node *l2 = make_list(a,5);

    print_list(l2);

    find_and_delete2(&l2, 3);

    print_list(l2);

    find_and_delete2(&l2, 1);

    print_list(l2);

    return 0;

}

{% endhighlight %}
