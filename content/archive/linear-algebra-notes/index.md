---

layout: post

title: "Linear Algebra Notes"

date: "2017-05-26 21:36:02 +0800"

---



## column space and null space

$$A$$ is a $$m \times n$$ matrix

$$R = rref(A) $$ with rank $$r$$



We have the following definitions:

column space of $$A$$:  $$ C(A) = \{ A\vec{x} | \vec{x} \in R^n \} $$

null space of $$A$$: $$N(A) = \{ \vec{x} | A\vec{x} = \vec{0} \}$$



then:

$$C(A)$$ = span(all pivot columns in A) (pivot columns in A means the corresponding columns to those pivot columns in R)



## Why the pivot columns in A are the basis of C(A) ?

first we need to prove that the pivot columns are linearly independent:

[https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/showing-relation-between-basis-cols-and-pivot-cols](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/showing-relation-between-basis-cols-and-pivot-cols)

then we can show that the free columns are linear combinations of the pivot columns:

[https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/showing-that-the-candidate-basis-does-span-c-a](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/null-column-space/v/showing-that-the-candidate-basis-does-span-c-a)



Thus we have:

$$dim(C(A))$$ = # of pivot variables = $$rank(A)$$



$$N(A) = N(R)$$ (because elimination does not change the solutions)



$$dim(N(A))$$ = # of free variables = $$n -$$ # of pivot variables = $$n - r$$



# Linear transformations



## visualization and intuition:

[https://www.khanacademy.org/math/linear-algebra/matrix-transformations/linear-transformations/a/visualizing-linear-transformations](https://www.khanacademy.org/math/linear-algebra/matrix-transformations/linear-transformations/a/visualizing-linear-transformations)



## linear transformation vs matrix-vector product

For any linear transformation $$T$$:

$$ T(\vec{x}) = T(I\vec{x}) = T(x_1\vec{e_1} + x_2\vec{e_2} + ... + x_n\vec{e_n}) = {x_1}T(\vec{e_1}) + {x_2}T(\vec{e_2}) + ... + {x_n}T(\vec{e_n}) = [T(\vec{e_1}) T(\vec{e_2}) ... T(\vec{e_n})]x $$

we have $$T(\vec{x}) = A\vec{x}$$, and the columns of $$A$$ are the results of applying the transformation on the standard basis $$(\vec{e_1}, \vec{e_2} ... \vec{e_n})$$

Hence, linear transformations are always 1 - 1 associated with matrix-vector product.



## definition: kernel of a linear transformation

Suppose we have a L.T $$T(\vec{x}) = A\vec{x}$$

Then we define $$kernel(T) = {\vec{x} | T(\vec{x}) = 0}$$, which is just null space of A



## composition of linear transformations

We have transformations: $$S(x) = A\vec{x}, T(\vec{x}) = Bx$$

define composition of $$S$$ and $$T$$: $$(S \circ T) (\vec{x}) = S(T(\vec{x}))$$

We can prove that $$(S \circ T)$$ is a linear transformation. So we have a matrix $$C$$, where $$(S \circ T) (\vec{x}) = C\vec{x}$$

To find $$C$$, we apply $$(S \circ T)$$ to $$I$$ :

$$C = S(T(I)) = A(B(I)) = \begin{bmatrix}A(B\vec{e_1}) & A(B\vec{e_2}) & A(B\vec{e_3}) & ... & A(B\vec{e_n}) \end{bmatrix} = \begin{bmatrix} A\vec{b_1} & A\vec{b_2} & A\vec{b_3} & ... & A\vec{b_n} \end{bmatrix}$$

This is the definition of $$C = A * B = \begin{bmatrix} A\vec{b_1} & A\vec{b_2} & A\vec{b_3} & ... & A\vec{b_n} \end{bmatrix}$$



## proof of matrix multiplication  associativity

Suppose $$H, G, F$$ are L.T.s, and $$A, B, C$$ are their corresponding matrices.

$$ H(x) = Ax, G(x) = Bx, F(x) = Cx $$

Then:

$$ ((AB)C) x = ((H \circ G) \circ F)(x) = ((H \circ G)(F(x)) = H(G(F(x))) = H((G \circ F)(x)) = (H \circ (G \circ F))(x) = (A(BC))x $$

thus we have  $$((AB)C) = (A(BC))$$

In short, matrix multiplications are associative because L.T.s(which are functions!) are associative.



## If A is invertible, then $$A^T A$$ is invertible.

proof:

$$A^T A \vec{v} = 0 \Rightarrow \vec{v}^TA^T A\vec{v} = 0 \Rightarrow ||Av||^2 = 0 \Rightarrow A\vec{v} = 0 $$

because $$A$$ is invertible, $$\vec{v}$$ must be 0.  This means $$A^TA$$ only has zero vector in the null space, and is thus invertible.



# Projections



## projection onto a subspace

Suppose we have a matrix $$A$$ and a vector $$\vec{x}$$, $$A\vec{y}$$ is the projection of $$\vec{x}$$ onto the column space of A.

By definition of projections, we have:

$$ A^T(\vec{x} - A\vec{y}) = 0 $$

So we can solve for y:

$$ \vec{y} = (A^TA)^{-1}A^T\vec{x} $$

And the projection will be:

$$ \vec{x_p} = A(A^TA)^{-1}A^T\vec{x} $$

This shows that projections are linear transformations.



## orthogonal complement

$$V$$ is a subspace of $$S$$, then any vector $$\vec{v} \in S$$ can be written as

$$ \vec{v} = \vec{p} + \vec{n} $$

where $$\vec{p} \in V$$ and $$\vec{n} \in V^\bot$$, $$V^\bot$$ being the orthogonal complement of $$V$$ in $$S$$:

$$V^\bot = \{\vec{x} | \vec{x}^T \vec{v} = 0 | \vec{v} \in V\}$$

Here, $$\vec{p}$$ is the projection of $$\vec{v}$$ onto $$V$$, and $$\vec{n}$$ is the projection of $$\vec{v}$$ onto $$V^\bot$$.

Suppose $$\vec{n} = B\vec{v}$$, then

$$\vec{v} = A(A^TA)^{-1}A^T\vec{v} + B\vec{v}$$

This way we can get the projection transformation matrix onto $$V^\bot$$:

$$B = I - A(A^TA)^{-1}A^T$$



## change of basis

suppose we have a basis $$B = \{\vec{v_1}, \vec{v_2}, ... \vec{v_n}\} $$ for $$R^n$$, then any vector $$\vec{v}$$ in $$R^n$$ can be represented by $$B$$ as

$$ \vec{v} = c_1\vec{v_1} + c_2\vec{v_2} + ... + c_n\vec{v_n} $$

Put in another way, $$\vec{v}$$ is $$\begin{bmatrix} c_1 \\ c_2 \\ ... \\c_n \end{bmatrix}$$ with respect to the new coordinate system given by $$B$$.

We give this new vector a notation as $$ {[\vec{v}]}_B $$, then:



$$ \vec{v} = C{[\vec{v}]}_B, C = \begin{bmatrix} \vec{v_1} & \vec{v_2} & ... & \vec{v_n} \end{bmatrix}$$

Where $$\vec{v_1}, \vec{v_2} ... \vec{v_n}$$ are the basis of $$B$$ under the standard coordinates. This is called the change of basis formula.



Now suppose we have a L.T. $$T(x) = Ax$$ in $$R^n$$, and under the coordinate system $$B$$, the same transformation is $${[T(x)]}_B = {[Ax]}_B $$. Suppose the new transformation matrix is $$D$$, then:

$$ {[T(x)]}_B = D{[x]}_B $$

so we have

$$ {[Ax]}_B = D{[x]}_B $$

Using the change of basis, we have

$$ C^{-1}Ax = DC^{-1}x  $$

so

$$ D = C^{-1}AC $$

This is also the definition that matrices $$A$$ and $$D$$ are similar.



## orthonormal basis

$$B = \{\vec{v_1}, \vec{v_2}, ... \vec{v_n}\} $$ for subspace $$V$$ of $$R^n$$ is _orthonormal_ iff:

$$||\vec{v_i}|| = 1$$

$$\vec{v_i} \cdot \vec{v_j} = 0, i \neq j $$



now suppose we have $$x \in V$$

$$ x = c_1v_1 + c_2v_2+ ... +c_nv_n $$

take dot product of both sides with $$v_i$$ we can get:

$$ c_i = v_i \cdot x $$

So the i-th component for vector $$x$$ with respect to $$B$$ is the dot product of the i-th basis and x.



## projection onto subspace with orthonormal basis

suppose $$x \in R^n$$, $$B = \{\vec{v_1}, \vec{v_2}, ... \vec{v_n}\} $$ is  an orthonormal basis for subspace $$V$$ of $$R^n$$, then $$x$$ can be represented as

$$x = c_1v_1 + c_2v_2+ ... +c_nv_n + w$$

where $$w \in V^\bot, w \cdot v_i = 0$$.

We take dot product of both sides by $$v_i$$:

$$c_i = v_i \cdot x$$



## orthonormal matrices transformations preserves lengths and angles

suppose $$C$$ is orthonormal, we can prove that

$$||Cx|| = ||x||$$

$$\frac{x \cdot y}{||x|| \cdot ||y||} = \frac{Cx \cdot Cy}{||Cx||\cdot||Cy||} $$



## Gram-Schmidt process

Converts any basis to orthonormal basis for the same subspace.

Starting from normalizing the first vector, subtract every vector with the projection onto the span of previous vectors. Examples:

[https://www.khanacademy.org/math/linear-algebra/alternate-bases/orthonormal-basis/v/linear-algebra-gram-schmidt-process-example](https://www.khanacademy.org/math/linear-algebra/alternate-bases/orthonormal-basis/v/linear-algebra-gram-schmidt-process-example)



## Eigenvalues and Eigenvectors

Eigenvectors of a linear transformation are the non-zero vectors that stay in the same direction (or flip around) when applying the linear transformation.

$$ Av = \lambda{v} $$

or:

$$ (\lambda{I} - A)v = 0 $$

because $$v$$ is not zero and $$v$$ is in the null space of $$\lambda{I} - A$$, $$\lambda{I} - A$$ must be singular.



When switching to the eigenvectors of transformation matrix $$A$$ as basis, the new transformation matrix will become a diagonal matrix with values $$\lambda_1, \lambda_2, ... \lambda_n$$.
