�9
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
�
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint���������"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
n
	ReverseV2
tensor"T
axis"Tidx
output"T"
Tidxtype0:
2	"
Ttype:
2	

l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
�
TopKV2

input"T
k"Tk
values"T
indices"
index_type"
sortedbool("
Ttype:
2	"
Tktype0:
2	"

index_typetype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��4
�
0bidirectional_93/backward_lstm_93/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20bidirectional_93/backward_lstm_93/lstm_cell/bias
�
Dbidirectional_93/backward_lstm_93/lstm_cell/bias/Read/ReadVariableOpReadVariableOp0bidirectional_93/backward_lstm_93/lstm_cell/bias*
_output_shapes
: *
dtype0
�
<bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel
�
Pbidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp<bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel*
_output_shapes

: *
dtype0
�
2bidirectional_93/backward_lstm_93/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *C
shared_name42bidirectional_93/backward_lstm_93/lstm_cell/kernel
�
Fbidirectional_93/backward_lstm_93/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp2bidirectional_93/backward_lstm_93/lstm_cell/kernel*
_output_shapes

: *
dtype0
�
/bidirectional_93/forward_lstm_93/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/bidirectional_93/forward_lstm_93/lstm_cell/bias
�
Cbidirectional_93/forward_lstm_93/lstm_cell/bias/Read/ReadVariableOpReadVariableOp/bidirectional_93/forward_lstm_93/lstm_cell/bias*
_output_shapes
: *
dtype0
�
;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *L
shared_name=;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel
�
Obidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel*
_output_shapes

: *
dtype0
�
1bidirectional_93/forward_lstm_93/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *B
shared_name31bidirectional_93/forward_lstm_93/lstm_cell/kernel
�
Ebidirectional_93/forward_lstm_93/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp1bidirectional_93/forward_lstm_93/lstm_cell/kernel*
_output_shapes

: *
dtype0
t
dense_610/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_610/bias
m
"dense_610/bias/Read/ReadVariableOpReadVariableOpdense_610/bias*
_output_shapes
:*
dtype0
|
dense_610/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_610/kernel
u
$dense_610/kernel/Read/ReadVariableOpReadVariableOpdense_610/kernel*
_output_shapes

:*
dtype0
t
dense_605/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_605/bias
m
"dense_605/bias/Read/ReadVariableOpReadVariableOpdense_605/bias*
_output_shapes
:*
dtype0
|
dense_605/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_605/kernel
u
$dense_605/kernel/Read/ReadVariableOpReadVariableOpdense_605/kernel*
_output_shapes

:*
dtype0
t
dense_600/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_600/bias
m
"dense_600/bias/Read/ReadVariableOpReadVariableOpdense_600/bias*
_output_shapes
:*
dtype0
|
dense_600/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_600/kernel
u
$dense_600/kernel/Read/ReadVariableOpReadVariableOpdense_600/kernel*
_output_shapes

:*
dtype0
t
dense_595/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_595/bias
m
"dense_595/bias/Read/ReadVariableOpReadVariableOpdense_595/bias*
_output_shapes
:*
dtype0
|
dense_595/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_595/kernel
u
$dense_595/kernel/Read/ReadVariableOpReadVariableOpdense_595/kernel*
_output_shapes

:*
dtype0
t
dense_611/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_611/bias
m
"dense_611/bias/Read/ReadVariableOpReadVariableOpdense_611/bias*
_output_shapes
:*
dtype0
}
dense_611/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*!
shared_namedense_611/kernel
v
$dense_611/kernel/Read/ReadVariableOpReadVariableOpdense_611/kernel*
_output_shapes
:	�*
dtype0
t
dense_594/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_594/bias
m
"dense_594/bias/Read/ReadVariableOpReadVariableOpdense_594/bias*
_output_shapes
:*
dtype0
|
dense_594/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_594/kernel
u
$dense_594/kernel/Read/ReadVariableOpReadVariableOpdense_594/kernel*
_output_shapes

:*
dtype0
p
serving_default_input_4Placeholder*"
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_594/kerneldense_594/biasdense_595/kerneldense_595/biasdense_600/kerneldense_600/biasdense_605/kerneldense_605/biasdense_610/kerneldense_610/bias1bidirectional_93/forward_lstm_93/lstm_cell/kernel;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel/bidirectional_93/forward_lstm_93/lstm_cell/bias2bidirectional_93/backward_lstm_93/lstm_cell/kernel<bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel0bidirectional_93/backward_lstm_93/lstm_cell/biasdense_611/kerneldense_611/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_55912384

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*̄
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures
#_self_saveable_object_factories*
'
#_self_saveable_object_factories* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
#$_self_saveable_object_factories*
6
%	keras_api
#&_self_saveable_object_factories* 
6
'	keras_api
#(_self_saveable_object_factories* 
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
#/_self_saveable_object_factories* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
#6_self_saveable_object_factories* 
6
7	keras_api
#8_self_saveable_object_factories* 
�
9layer_with_weights-0
9layer-0
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
#@_self_saveable_object_factories*
�
Alayer_with_weights-0
Alayer-0
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
#H_self_saveable_object_factories*
�
Ilayer_with_weights-0
Ilayer-0
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
#P_self_saveable_object_factories*
�
Qlayer_with_weights-0
Qlayer-0
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
#X_self_saveable_object_factories*
6
Y	keras_api
#Z_self_saveable_object_factories* 
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
#a_self_saveable_object_factories* 
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
hforward_layer
ibackward_layer
#j_self_saveable_object_factories*
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
q_random_generator
#r_self_saveable_object_factories* 
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
#y_self_saveable_object_factories* 
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories*
�
"0
#1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
�
"0
#1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�serving_default* 
* 
* 

"0
#1*

"0
#1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_594/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_594/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
4
�0
�1
�2
�3
�4
�5*
4
�0
�1
�2
�3
�4
�5*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
	�cell
�
state_spec
$�_self_saveable_object_factories*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
	�cell
�
state_spec
$�_self_saveable_object_factories*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
(
$�_self_saveable_object_factories* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEdense_611/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_611/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
PJ
VARIABLE_VALUEdense_595/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_595/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_600/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_600/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_605/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_605/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_610/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_610/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1bidirectional_93/forward_lstm_93/lstm_cell/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/bidirectional_93/forward_lstm_93/lstm_cell/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2bidirectional_93/backward_lstm_93/lstm_cell/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0bidirectional_93/backward_lstm_93/lstm_cell/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

90*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

A0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

I0*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 

Q0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

h0
i1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1
�2*

�0
�1
�2*
* 
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
(
$�_self_saveable_object_factories* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�
state_size
�kernel
�recurrent_kernel
	�bias
$�_self_saveable_object_factories*
* 
* 

�0
�1
�2*

�0
�1
�2*
* 
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
(
$�_self_saveable_object_factories* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�
state_size
�kernel
�recurrent_kernel
	�bias
$�_self_saveable_object_factories*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1
�2*

�0
�1
�2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
(
$�_self_saveable_object_factories* 
* 
* 
* 
* 

�0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1
�2*

�0
�1
�2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
(
$�_self_saveable_object_factories* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_594/kerneldense_594/biasdense_611/kerneldense_611/biasdense_595/kerneldense_595/biasdense_600/kerneldense_600/biasdense_605/kerneldense_605/biasdense_610/kerneldense_610/bias1bidirectional_93/forward_lstm_93/lstm_cell/kernel;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel/bidirectional_93/forward_lstm_93/lstm_cell/bias2bidirectional_93/backward_lstm_93/lstm_cell/kernel<bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel0bidirectional_93/backward_lstm_93/lstm_cell/biasConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_55915483
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_594/kerneldense_594/biasdense_611/kerneldense_611/biasdense_595/kerneldense_595/biasdense_600/kerneldense_600/biasdense_605/kerneldense_605/biasdense_610/kerneldense_610/bias1bidirectional_93/forward_lstm_93/lstm_cell/kernel;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel/bidirectional_93/forward_lstm_93/lstm_cell/bias2bidirectional_93/backward_lstm_93/lstm_cell/kernel<bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel0bidirectional_93/backward_lstm_93/lstm_cell/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_55915546��2
�	
�
G__inference_dense_611_layer_call_and_return_conditional_losses_55911775

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	�
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
,__inference_dense_610_layer_call_fn_55913886

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_610_layer_call_and_return_conditional_losses_55909932s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
55913880:($
"
_user_specified_name
55913882
�	
�
while_cond_55910050
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55910050___redundant_placeholder06
2while_while_cond_55910050___redundant_placeholder16
2while_while_cond_55910050___redundant_placeholder26
2while_while_cond_55910050___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�M
�
$backward_lstm_93_while_body_55912748>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3=
9backward_lstm_93_while_backward_lstm_93_strided_slice_1_0y
ubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_93_while_identity%
!backward_lstm_93_while_identity_1%
!backward_lstm_93_while_identity_2%
!backward_lstm_93_while_identity_3%
!backward_lstm_93_while_identity_4%
!backward_lstm_93_while_identity_5;
7backward_lstm_93_while_backward_lstm_93_strided_slice_1w
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Hbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
:backward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_93_while_placeholderQbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
'backward_lstm_93/while/lstm_cell/MatMulMatMulAbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
)backward_lstm_93/while/lstm_cell/MatMul_1MatMul$backward_lstm_93_while_placeholder_2@backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
$backward_lstm_93/while/lstm_cell/addAddV21backward_lstm_93/while/lstm_cell/MatMul:product:03backward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
(backward_lstm_93/while/lstm_cell/BiasAddBiasAdd(backward_lstm_93/while/lstm_cell/add:z:0?backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
0backward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
&backward_lstm_93/while/lstm_cell/splitSplit9backward_lstm_93/while/lstm_cell/split/split_dim:output:01backward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
(backward_lstm_93/while/lstm_cell/SigmoidSigmoid/backward_lstm_93/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:����������
*backward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_93/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
$backward_lstm_93/while/lstm_cell/mulMul.backward_lstm_93/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_93_while_placeholder_3*
T0*'
_output_shapes
:����������
%backward_lstm_93/while/lstm_cell/ReluRelu/backward_lstm_93/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
&backward_lstm_93/while/lstm_cell/mul_1Mul,backward_lstm_93/while/lstm_cell/Sigmoid:y:03backward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
&backward_lstm_93/while/lstm_cell/add_1AddV2(backward_lstm_93/while/lstm_cell/mul:z:0*backward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:����������
*backward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_93/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:����������
'backward_lstm_93/while/lstm_cell/Relu_1Relu*backward_lstm_93/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
&backward_lstm_93/while/lstm_cell/mul_2Mul.backward_lstm_93/while/lstm_cell/Sigmoid_2:y:05backward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
;backward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_93_while_placeholder_1"backward_lstm_93_while_placeholder*backward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���^
backward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/addAddV2"backward_lstm_93_while_placeholder%backward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/add_1AddV2:backward_lstm_93_while_backward_lstm_93_while_loop_counter'backward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
backward_lstm_93/while/IdentityIdentity backward_lstm_93/while/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_1Identity@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_2Identitybackward_lstm_93/while/add:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_3IdentityKbackward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_4Identity*backward_lstm_93/while/lstm_cell/mul_2:z:0^backward_lstm_93/while/NoOp*
T0*'
_output_shapes
:����������
!backward_lstm_93/while/Identity_5Identity*backward_lstm_93/while/lstm_cell/add_1:z:0^backward_lstm_93/while/NoOp*
T0*'
_output_shapes
:����������
backward_lstm_93/while/NoOpNoOp8^backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_93_while_backward_lstm_93_strided_slice_19backward_lstm_93_while_backward_lstm_93_strided_slice_1_0"K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0"O
!backward_lstm_93_while_identity_1*backward_lstm_93/while/Identity_1:output:0"O
!backward_lstm_93_while_identity_2*backward_lstm_93/while/Identity_2:output:0"O
!backward_lstm_93_while_identity_3*backward_lstm_93/while/Identity_3:output:0"O
!backward_lstm_93_while_identity_4*backward_lstm_93/while/Identity_4:output:0"O
!backward_lstm_93_while_identity_5*backward_lstm_93/while/Identity_5:output:0"�
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2r
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
��
�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55912836
inputs_0J
8forward_lstm_93_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_93_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource: 
identity��1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp�2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�backward_lstm_93/while�0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp�1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�forward_lstm_93/while[
forward_lstm_93/ShapeShapeinputs_0*
T0*
_output_shapes
::��m
#forward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_sliceStridedSliceforward_lstm_93/Shape:output:0,forward_lstm_93/strided_slice/stack:output:0.forward_lstm_93/strided_slice/stack_1:output:0.forward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros/packedPack&forward_lstm_93/strided_slice:output:0'forward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zerosFill%forward_lstm_93/zeros/packed:output:0$forward_lstm_93/zeros/Const:output:0*
T0*'
_output_shapes
:���������b
 forward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros_1/packedPack&forward_lstm_93/strided_slice:output:0)forward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zeros_1Fill'forward_lstm_93/zeros_1/packed:output:0&forward_lstm_93/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������s
forward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose	Transposeinputs_0'forward_lstm_93/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������r
forward_lstm_93/Shape_1Shapeforward_lstm_93/transpose:y:0*
T0*
_output_shapes
::��o
%forward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_1StridedSlice forward_lstm_93/Shape_1:output:0.forward_lstm_93/strided_slice_1/stack:output:00forward_lstm_93/strided_slice_1/stack_1:output:00forward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
forward_lstm_93/TensorArrayV2TensorListReserve4forward_lstm_93/TensorArrayV2/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Eforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
7forward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_93/transpose:y:0Nforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���o
%forward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_2StridedSliceforward_lstm_93/transpose:y:0.forward_lstm_93/strided_slice_2/stack:output:00forward_lstm_93/strided_slice_2/stack_1:output:00forward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 forward_lstm_93/lstm_cell/MatMulMatMul(forward_lstm_93/strided_slice_2:output:07forward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
"forward_lstm_93/lstm_cell/MatMul_1MatMulforward_lstm_93/zeros:output:09forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
forward_lstm_93/lstm_cell/addAddV2*forward_lstm_93/lstm_cell/MatMul:product:0,forward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!forward_lstm_93/lstm_cell/BiasAddBiasAdd!forward_lstm_93/lstm_cell/add:z:08forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� k
)forward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/lstm_cell/splitSplit2forward_lstm_93/lstm_cell/split/split_dim:output:0*forward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
!forward_lstm_93/lstm_cell/SigmoidSigmoid(forward_lstm_93/lstm_cell/split:output:0*
T0*'
_output_shapes
:����������
#forward_lstm_93/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_93/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/mulMul'forward_lstm_93/lstm_cell/Sigmoid_1:y:0 forward_lstm_93/zeros_1:output:0*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/ReluRelu(forward_lstm_93/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/mul_1Mul%forward_lstm_93/lstm_cell/Sigmoid:y:0,forward_lstm_93/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/add_1AddV2!forward_lstm_93/lstm_cell/mul:z:0#forward_lstm_93/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:����������
#forward_lstm_93/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_93/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������
 forward_lstm_93/lstm_cell/Relu_1Relu#forward_lstm_93/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/mul_2Mul'forward_lstm_93/lstm_cell/Sigmoid_2:y:0.forward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������~
-forward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
forward_lstm_93/TensorArrayV2_1TensorListReserve6forward_lstm_93/TensorArrayV2_1/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���V
forward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������d
"forward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
forward_lstm_93/whileWhile+forward_lstm_93/while/loop_counter:output:01forward_lstm_93/while/maximum_iterations:output:0forward_lstm_93/time:output:0(forward_lstm_93/TensorArrayV2_1:handle:0forward_lstm_93/zeros:output:0 forward_lstm_93/zeros_1:output:0(forward_lstm_93/strided_slice_1:output:0Gforward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_93_lstm_cell_matmul_readvariableop_resource:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#forward_lstm_93_while_body_55912607*/
cond'R%
#forward_lstm_93_while_cond_55912606*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
@forward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2forward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_93/while:output:3Iforward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0x
%forward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������q
'forward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_3StridedSlice;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_93/strided_slice_3/stack:output:00forward_lstm_93/strided_slice_3/stack_1:output:00forward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masku
 forward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose_1	Transpose;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_93/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
forward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
backward_lstm_93/ShapeShapeinputs_0*
T0*
_output_shapes
::��n
$backward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
backward_lstm_93/strided_sliceStridedSlicebackward_lstm_93/Shape:output:0-backward_lstm_93/strided_slice/stack:output:0/backward_lstm_93/strided_slice/stack_1:output:0/backward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros/packedPack'backward_lstm_93/strided_slice:output:0(backward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zerosFill&backward_lstm_93/zeros/packed:output:0%backward_lstm_93/zeros/Const:output:0*
T0*'
_output_shapes
:���������c
!backward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros_1/packedPack'backward_lstm_93/strided_slice:output:0*backward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zeros_1Fill(backward_lstm_93/zeros_1/packed:output:0'backward_lstm_93/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������t
backward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose	Transposeinputs_0(backward_lstm_93/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������t
backward_lstm_93/Shape_1Shapebackward_lstm_93/transpose:y:0*
T0*
_output_shapes
::��p
&backward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_1StridedSlice!backward_lstm_93/Shape_1:output:0/backward_lstm_93/strided_slice_1/stack:output:01backward_lstm_93/strided_slice_1/stack_1:output:01backward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
backward_lstm_93/TensorArrayV2TensorListReserve5backward_lstm_93/TensorArrayV2/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���i
backward_lstm_93/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
backward_lstm_93/ReverseV2	ReverseV2backward_lstm_93/transpose:y:0(backward_lstm_93/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'����������������������������
Fbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
8backward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_93/ReverseV2:output:0Obackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���p
&backward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_2StridedSlicebackward_lstm_93/transpose:y:0/backward_lstm_93/strided_slice_2/stack:output:01backward_lstm_93/strided_slice_2/stack_1:output:01backward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
!backward_lstm_93/lstm_cell/MatMulMatMul)backward_lstm_93/strided_slice_2:output:08backward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
#backward_lstm_93/lstm_cell/MatMul_1MatMulbackward_lstm_93/zeros:output:0:backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
backward_lstm_93/lstm_cell/addAddV2+backward_lstm_93/lstm_cell/MatMul:product:0-backward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"backward_lstm_93/lstm_cell/BiasAddBiasAdd"backward_lstm_93/lstm_cell/add:z:09backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� l
*backward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 backward_lstm_93/lstm_cell/splitSplit3backward_lstm_93/lstm_cell/split/split_dim:output:0+backward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
"backward_lstm_93/lstm_cell/SigmoidSigmoid)backward_lstm_93/lstm_cell/split:output:0*
T0*'
_output_shapes
:����������
$backward_lstm_93/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_93/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
backward_lstm_93/lstm_cell/mulMul(backward_lstm_93/lstm_cell/Sigmoid_1:y:0!backward_lstm_93/zeros_1:output:0*
T0*'
_output_shapes
:����������
backward_lstm_93/lstm_cell/ReluRelu)backward_lstm_93/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
 backward_lstm_93/lstm_cell/mul_1Mul&backward_lstm_93/lstm_cell/Sigmoid:y:0-backward_lstm_93/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
 backward_lstm_93/lstm_cell/add_1AddV2"backward_lstm_93/lstm_cell/mul:z:0$backward_lstm_93/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:����������
$backward_lstm_93/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_93/lstm_cell/split:output:3*
T0*'
_output_shapes
:����������
!backward_lstm_93/lstm_cell/Relu_1Relu$backward_lstm_93/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
 backward_lstm_93/lstm_cell/mul_2Mul(backward_lstm_93/lstm_cell/Sigmoid_2:y:0/backward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������
.backward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
 backward_lstm_93/TensorArrayV2_1TensorListReserve7backward_lstm_93/TensorArrayV2_1/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���W
backward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������e
#backward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
backward_lstm_93/whileWhile,backward_lstm_93/while/loop_counter:output:02backward_lstm_93/while/maximum_iterations:output:0backward_lstm_93/time:output:0)backward_lstm_93/TensorArrayV2_1:handle:0backward_lstm_93/zeros:output:0!backward_lstm_93/zeros_1:output:0)backward_lstm_93/strided_slice_1:output:0Hbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_93_lstm_cell_matmul_readvariableop_resource;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$backward_lstm_93_while_body_55912748*0
cond(R&
$backward_lstm_93_while_cond_55912747*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
Abackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
3backward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_93/while:output:3Jbackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0y
&backward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������r
(backward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_3StridedSlice<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_93/strided_slice_3/stack:output:01backward_lstm_93/strided_slice_3/stack_1:output:01backward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskv
!backward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose_1	Transpose<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_93/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������l
backward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
	ReverseV2	ReverseV2 backward_lstm_93/transpose_1:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2forward_lstm_93/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :������������������k
IdentityIdentityconcat:output:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp2^backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_93/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_93/while1^forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_93/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2f
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_93/whilebackward_lstm_93/while2d
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_93/whileforward_lstm_93/while:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
t
H__inference_lambda_101_layer_call_and_return_conditional_losses_55912480
inputs_0
inputs_1
identity~
einsum/EinsumEinsuminputs_0inputs_1*
N*
T0*"
_output_shapes
:*
equationabcd,ace->acdY
IdentityIdentityeinsum/Einsum:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::P L
&
_output_shapes
:
"
_user_specified_name
inputs_0:LH
"
_output_shapes
:
"
_user_specified_name
inputs_1
�
�
#forward_lstm_93_while_cond_55913182<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3>
:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55913182___redundant_placeholder0V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55913182___redundant_placeholder1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55913182___redundant_placeholder2V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55913182___redundant_placeholder3"
forward_lstm_93_while_identity
�
forward_lstm_93/while/LessLess!forward_lstm_93_while_placeholder:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:

_output_shapes
:
�$
�
while_body_55910051
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_55910075_0: ,
while_lstm_cell_55910077_0: (
while_lstm_cell_55910079_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_55910075: *
while_lstm_cell_55910077: &
while_lstm_cell_55910079: ��'while/lstm_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55910075_0while_lstm_cell_55910077_0while_lstm_cell_55910079_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910037�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_55910075while_lstm_cell_55910075_0"6
while_lstm_cell_55910077while_lstm_cell_55910077_0"6
while_lstm_cell_55910079while_lstm_cell_55910079_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
55910075:(	$
"
_user_specified_name
55910077:(
$
"
_user_specified_name
55910079
�
�
1__inference_sequential_418_layer_call_fn_55909890
dense_605_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_605_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909872s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_605_input:($
"
_user_specified_name
55909884:($
"
_user_specified_name
55909886
�Y
�
$__inference__traced_restore_55915546
file_prefix3
!assignvariableop_dense_594_kernel:/
!assignvariableop_1_dense_594_bias:6
#assignvariableop_2_dense_611_kernel:	�/
!assignvariableop_3_dense_611_bias:5
#assignvariableop_4_dense_595_kernel:/
!assignvariableop_5_dense_595_bias:5
#assignvariableop_6_dense_600_kernel:/
!assignvariableop_7_dense_600_bias:5
#assignvariableop_8_dense_605_kernel:/
!assignvariableop_9_dense_605_bias:6
$assignvariableop_10_dense_610_kernel:0
"assignvariableop_11_dense_610_bias:W
Eassignvariableop_12_bidirectional_93_forward_lstm_93_lstm_cell_kernel: a
Oassignvariableop_13_bidirectional_93_forward_lstm_93_lstm_cell_recurrent_kernel: Q
Cassignvariableop_14_bidirectional_93_forward_lstm_93_lstm_cell_bias: X
Fassignvariableop_15_bidirectional_93_backward_lstm_93_lstm_cell_kernel: b
Passignvariableop_16_bidirectional_93_backward_lstm_93_lstm_cell_recurrent_kernel: R
Dassignvariableop_17_bidirectional_93_backward_lstm_93_lstm_cell_bias: 
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_dense_594_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_594_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_611_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_611_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_595_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_595_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_600_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_600_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_605_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_605_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_610_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_610_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpEassignvariableop_12_bidirectional_93_forward_lstm_93_lstm_cell_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpOassignvariableop_13_bidirectional_93_forward_lstm_93_lstm_cell_recurrent_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpCassignvariableop_14_bidirectional_93_forward_lstm_93_lstm_cell_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpFassignvariableop_15_bidirectional_93_backward_lstm_93_lstm_cell_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpPassignvariableop_16_bidirectional_93_backward_lstm_93_lstm_cell_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpDassignvariableop_17_bidirectional_93_backward_lstm_93_lstm_cell_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_19Identity_19:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_user_specified_namedense_594/kernel:.*
(
_user_specified_namedense_594/bias:0,
*
_user_specified_namedense_611/kernel:.*
(
_user_specified_namedense_611/bias:0,
*
_user_specified_namedense_595/kernel:.*
(
_user_specified_namedense_595/bias:0,
*
_user_specified_namedense_600/kernel:.*
(
_user_specified_namedense_600/bias:0	,
*
_user_specified_namedense_605/kernel:.
*
(
_user_specified_namedense_605/bias:0,
*
_user_specified_namedense_610/kernel:.*
(
_user_specified_namedense_610/bias:QM
K
_user_specified_name31bidirectional_93/forward_lstm_93/lstm_cell/kernel:[W
U
_user_specified_name=;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel:OK
I
_user_specified_name1/bidirectional_93/forward_lstm_93/lstm_cell/bias:RN
L
_user_specified_name42bidirectional_93/backward_lstm_93/lstm_cell/kernel:\X
V
_user_specified_name><bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel:PL
J
_user_specified_name20bidirectional_93/backward_lstm_93/lstm_cell/bias
�
�
,__inference_dense_611_layer_call_fn_55913747

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_611_layer_call_and_return_conditional_losses_55911775f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	�: : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	�
 
_user_specified_nameinputs:($
"
_user_specified_name
55913741:($
"
_user_specified_name
55913743
�
�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915353

inputs
states_0
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910532

inputs

states
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
1__inference_sequential_418_layer_call_fn_55909881
dense_605_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_605_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909863s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_605_input:($
"
_user_specified_name
55909875:($
"
_user_specified_name
55909877
�8
�
while_body_55910732
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
q
G__inference_lambda_99_layer_call_and_return_conditional_losses_55911404

inputs
inputs_1
identity|
einsum/EinsumEinsuminputsinputs_1*
N*
T0*"
_output_shapes
:*
equationabc,abcd->abdY
IdentityIdentityeinsum/Einsum:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::J F
"
_output_shapes
:
 
_user_specified_nameinputs:NJ
&
_output_shapes
:
 
_user_specified_nameinputs
�	
�
while_cond_55914637
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55914637___redundant_placeholder06
2while_while_cond_55914637___redundant_placeholder16
2while_while_cond_55914637___redundant_placeholder26
2while_while_cond_55914637___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�	
�
while_cond_55911193
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55911193___redundant_placeholder06
2while_while_cond_55911193___redundant_placeholder16
2while_while_cond_55911193___redundant_placeholder26
2while_while_cond_55911193___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�J
�
#forward_lstm_93_while_body_55911901<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3;
7forward_lstm_93_while_forward_lstm_93_strided_slice_1_0w
sforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_93_while_identity$
 forward_lstm_93_while_identity_1$
 forward_lstm_93_while_identity_2$
 forward_lstm_93_while_identity_3$
 forward_lstm_93_while_identity_4$
 forward_lstm_93_while_identity_59
5forward_lstm_93_while_forward_lstm_93_strided_slice_1u
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Gforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
9forward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_93_while_placeholderPforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
&forward_lstm_93/while/lstm_cell/MatMulMatMul@forward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
(forward_lstm_93/while/lstm_cell/MatMul_1MatMul#forward_lstm_93_while_placeholder_2?forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
#forward_lstm_93/while/lstm_cell/addAddV20forward_lstm_93/while/lstm_cell/MatMul:product:02forward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
'forward_lstm_93/while/lstm_cell/BiasAddBiasAdd'forward_lstm_93/while/lstm_cell/add:z:0>forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
/forward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
%forward_lstm_93/while/lstm_cell/splitSplit8forward_lstm_93/while/lstm_cell/split/split_dim:output:00forward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
'forward_lstm_93/while/lstm_cell/SigmoidSigmoid.forward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
)forward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
#forward_lstm_93/while/lstm_cell/mulMul-forward_lstm_93/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
$forward_lstm_93/while/lstm_cell/ReluRelu.forward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/mul_1Mul+forward_lstm_93/while/lstm_cell/Sigmoid:y:02forward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/add_1AddV2'forward_lstm_93/while/lstm_cell/mul:z:0)forward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
)forward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
&forward_lstm_93/while/lstm_cell/Relu_1Relu)forward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/mul_2Mul-forward_lstm_93/while/lstm_cell/Sigmoid_2:y:04forward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
:forward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_93_while_placeholder_1!forward_lstm_93_while_placeholder)forward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���]
forward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/addAddV2!forward_lstm_93_while_placeholder$forward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/add_1AddV28forward_lstm_93_while_forward_lstm_93_while_loop_counter&forward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_1Identity>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_2Identityforward_lstm_93/while/add:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_3IdentityJforward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_4Identity)forward_lstm_93/while/lstm_cell/mul_2:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
 forward_lstm_93/while/Identity_5Identity)forward_lstm_93/while/lstm_cell/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
forward_lstm_93/while/NoOpNoOp7^forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_93_while_forward_lstm_93_strided_slice_17forward_lstm_93_while_forward_lstm_93_strided_slice_1_0"I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0"M
 forward_lstm_93_while_identity_1)forward_lstm_93/while/Identity_1:output:0"M
 forward_lstm_93_while_identity_2)forward_lstm_93/while/Identity_2:output:0"M
 forward_lstm_93_while_identity_3)forward_lstm_93/while/Identity_3:output:0"M
 forward_lstm_93_while_identity_4)forward_lstm_93/while/Identity_4:output:0"M
 forward_lstm_93_while_identity_5)forward_lstm_93/while/Identity_5:output:0"�
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2p
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�	
�
while_cond_55914305
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55914305___redundant_placeholder06
2while_while_cond_55914305___redundant_placeholder16
2while_while_cond_55914305___redundant_placeholder26
2while_while_cond_55914305___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�K
�
$backward_lstm_93_while_body_55913324>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3=
9backward_lstm_93_while_backward_lstm_93_strided_slice_1_0y
ubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_93_while_identity%
!backward_lstm_93_while_identity_1%
!backward_lstm_93_while_identity_2%
!backward_lstm_93_while_identity_3%
!backward_lstm_93_while_identity_4%
!backward_lstm_93_while_identity_5;
7backward_lstm_93_while_backward_lstm_93_strided_slice_1w
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Hbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
:backward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_93_while_placeholderQbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
'backward_lstm_93/while/lstm_cell/MatMulMatMulAbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
)backward_lstm_93/while/lstm_cell/MatMul_1MatMul$backward_lstm_93_while_placeholder_2@backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
$backward_lstm_93/while/lstm_cell/addAddV21backward_lstm_93/while/lstm_cell/MatMul:product:03backward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
(backward_lstm_93/while/lstm_cell/BiasAddBiasAdd(backward_lstm_93/while/lstm_cell/add:z:0?backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: r
0backward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
&backward_lstm_93/while/lstm_cell/splitSplit9backward_lstm_93/while/lstm_cell/split/split_dim:output:01backward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
(backward_lstm_93/while/lstm_cell/SigmoidSigmoid/backward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
*backward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
$backward_lstm_93/while/lstm_cell/mulMul.backward_lstm_93/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
%backward_lstm_93/while/lstm_cell/ReluRelu/backward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/mul_1Mul,backward_lstm_93/while/lstm_cell/Sigmoid:y:03backward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/add_1AddV2(backward_lstm_93/while/lstm_cell/mul:z:0*backward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
*backward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
'backward_lstm_93/while/lstm_cell/Relu_1Relu*backward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/mul_2Mul.backward_lstm_93/while/lstm_cell/Sigmoid_2:y:05backward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
;backward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_93_while_placeholder_1"backward_lstm_93_while_placeholder*backward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���^
backward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/addAddV2"backward_lstm_93_while_placeholder%backward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/add_1AddV2:backward_lstm_93_while_backward_lstm_93_while_loop_counter'backward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
backward_lstm_93/while/IdentityIdentity backward_lstm_93/while/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_1Identity@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_2Identitybackward_lstm_93/while/add:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_3IdentityKbackward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_4Identity*backward_lstm_93/while/lstm_cell/mul_2:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
!backward_lstm_93/while/Identity_5Identity*backward_lstm_93/while/lstm_cell/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
backward_lstm_93/while/NoOpNoOp8^backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_93_while_backward_lstm_93_strided_slice_19backward_lstm_93_while_backward_lstm_93_strided_slice_1_0"K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0"O
!backward_lstm_93_while_identity_1*backward_lstm_93/while/Identity_1:output:0"O
!backward_lstm_93_while_identity_2*backward_lstm_93/while/Identity_2:output:0"O
!backward_lstm_93_while_identity_3*backward_lstm_93/while/Identity_3:output:0"O
!backward_lstm_93_while_identity_4*backward_lstm_93/while/Identity_4:output:0"O
!backward_lstm_93_while_identity_5*backward_lstm_93/while/Identity_5:output:0"�
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2r
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
1__inference_sequential_423_layer_call_fn_55909957
dense_610_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_610_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909939s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_610_input:($
"
_user_specified_name
55909951:($
"
_user_specified_name
55909953
�
�
#forward_lstm_93_while_cond_55912606<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3>
:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55912606___redundant_placeholder0V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55912606___redundant_placeholder1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55912606___redundant_placeholder2V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55912606___redundant_placeholder3"
forward_lstm_93_while_identity
�
forward_lstm_93/while/LessLess!forward_lstm_93_while_placeholder:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:

_output_shapes
:
�8
�
while_body_55911042
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�	
�
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909711
dense_595_input$
dense_595_55909705: 
dense_595_55909707:
identity��!dense_595/StatefulPartitionedCall�
!dense_595/StatefulPartitionedCallStatefulPartitionedCalldense_595_inputdense_595_55909705dense_595_55909707*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_595_layer_call_and_return_conditional_losses_55909704}
IdentityIdentity*dense_595/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������F
NoOpNoOp"^dense_595/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2F
!dense_595/StatefulPartitionedCall!dense_595/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_595_input:($
"
_user_specified_name
55909705:($
"
_user_specified_name
55909707
�
�
,__inference_dense_595_layer_call_fn_55913766

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_595_layer_call_and_return_conditional_losses_55909704s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
55913760:($
"
_user_specified_name
55913762
�
�
#forward_lstm_93_while_cond_55913470<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3>
:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55913470___redundant_placeholder0V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55913470___redundant_placeholder1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55913470___redundant_placeholder2V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55913470___redundant_placeholder3"
forward_lstm_93_while_identity
�
forward_lstm_93/while/LessLess!forward_lstm_93_while_placeholder:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:

_output_shapes
:
�
f
H__inference_dropout_93_layer_call_and_return_conditional_losses_55913727

inputs

identity_1I
IdentityIdentityinputs*
T0*"
_output_shapes
:V

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
:"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
t
H__inference_lambda_101_layer_call_and_return_conditional_losses_55912474
inputs_0
inputs_1
identity~
einsum/EinsumEinsuminputs_0inputs_1*
N*
T0*"
_output_shapes
:*
equationabcd,ace->acdY
IdentityIdentityeinsum/Einsum:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::P L
&
_output_shapes
:
"
_user_specified_name
inputs_0:LH
"
_output_shapes
:
"
_user_specified_name
inputs_1
�	
�
while_cond_55910883
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55910883___redundant_placeholder06
2while_while_cond_55910883___redundant_placeholder16
2while_while_cond_55910883___redundant_placeholder26
2while_while_cond_55910883___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�L
�
#forward_lstm_93_while_body_55912607<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3;
7forward_lstm_93_while_forward_lstm_93_strided_slice_1_0w
sforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_93_while_identity$
 forward_lstm_93_while_identity_1$
 forward_lstm_93_while_identity_2$
 forward_lstm_93_while_identity_3$
 forward_lstm_93_while_identity_4$
 forward_lstm_93_while_identity_59
5forward_lstm_93_while_forward_lstm_93_strided_slice_1u
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Gforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
9forward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_93_while_placeholderPforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
&forward_lstm_93/while/lstm_cell/MatMulMatMul@forward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
(forward_lstm_93/while/lstm_cell/MatMul_1MatMul#forward_lstm_93_while_placeholder_2?forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
#forward_lstm_93/while/lstm_cell/addAddV20forward_lstm_93/while/lstm_cell/MatMul:product:02forward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
'forward_lstm_93/while/lstm_cell/BiasAddBiasAdd'forward_lstm_93/while/lstm_cell/add:z:0>forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� q
/forward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
%forward_lstm_93/while/lstm_cell/splitSplit8forward_lstm_93/while/lstm_cell/split/split_dim:output:00forward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
'forward_lstm_93/while/lstm_cell/SigmoidSigmoid.forward_lstm_93/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:����������
)forward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_93/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
#forward_lstm_93/while/lstm_cell/mulMul-forward_lstm_93/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_93_while_placeholder_3*
T0*'
_output_shapes
:����������
$forward_lstm_93/while/lstm_cell/ReluRelu.forward_lstm_93/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
%forward_lstm_93/while/lstm_cell/mul_1Mul+forward_lstm_93/while/lstm_cell/Sigmoid:y:02forward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
%forward_lstm_93/while/lstm_cell/add_1AddV2'forward_lstm_93/while/lstm_cell/mul:z:0)forward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:����������
)forward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_93/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:����������
&forward_lstm_93/while/lstm_cell/Relu_1Relu)forward_lstm_93/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
%forward_lstm_93/while/lstm_cell/mul_2Mul-forward_lstm_93/while/lstm_cell/Sigmoid_2:y:04forward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
:forward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_93_while_placeholder_1!forward_lstm_93_while_placeholder)forward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���]
forward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/addAddV2!forward_lstm_93_while_placeholder$forward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/add_1AddV28forward_lstm_93_while_forward_lstm_93_while_loop_counter&forward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_1Identity>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_2Identityforward_lstm_93/while/add:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_3IdentityJforward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_4Identity)forward_lstm_93/while/lstm_cell/mul_2:z:0^forward_lstm_93/while/NoOp*
T0*'
_output_shapes
:����������
 forward_lstm_93/while/Identity_5Identity)forward_lstm_93/while/lstm_cell/add_1:z:0^forward_lstm_93/while/NoOp*
T0*'
_output_shapes
:����������
forward_lstm_93/while/NoOpNoOp7^forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_93_while_forward_lstm_93_strided_slice_17forward_lstm_93_while_forward_lstm_93_strided_slice_1_0"I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0"M
 forward_lstm_93_while_identity_1)forward_lstm_93/while/Identity_1:output:0"M
 forward_lstm_93_while_identity_2)forward_lstm_93/while/Identity_2:output:0"M
 forward_lstm_93_while_identity_3)forward_lstm_93/while/Identity_3:output:0"M
 forward_lstm_93_while_identity_4)forward_lstm_93/while/Identity_4:output:0"M
 forward_lstm_93_while_identity_5)forward_lstm_93/while/Identity_5:output:0"�
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2p
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
G__inference_dense_610_layer_call_and_return_conditional_losses_55913917

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
G__inference_dense_605_layer_call_and_return_conditional_losses_55909856

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
,__inference_lstm_cell_layer_call_fn_55915272

inputs
states_0
states_1
unknown: 
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910385o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1:($
"
_user_specified_name
55915260:($
"
_user_specified_name
55915262:($
"
_user_specified_name
55915264
�K
�
$backward_lstm_93_while_body_55911644>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3=
9backward_lstm_93_while_backward_lstm_93_strided_slice_1_0y
ubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_93_while_identity%
!backward_lstm_93_while_identity_1%
!backward_lstm_93_while_identity_2%
!backward_lstm_93_while_identity_3%
!backward_lstm_93_while_identity_4%
!backward_lstm_93_while_identity_5;
7backward_lstm_93_while_backward_lstm_93_strided_slice_1w
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Hbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
:backward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_93_while_placeholderQbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
'backward_lstm_93/while/lstm_cell/MatMulMatMulAbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
)backward_lstm_93/while/lstm_cell/MatMul_1MatMul$backward_lstm_93_while_placeholder_2@backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
$backward_lstm_93/while/lstm_cell/addAddV21backward_lstm_93/while/lstm_cell/MatMul:product:03backward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
(backward_lstm_93/while/lstm_cell/BiasAddBiasAdd(backward_lstm_93/while/lstm_cell/add:z:0?backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: r
0backward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
&backward_lstm_93/while/lstm_cell/splitSplit9backward_lstm_93/while/lstm_cell/split/split_dim:output:01backward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
(backward_lstm_93/while/lstm_cell/SigmoidSigmoid/backward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
*backward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
$backward_lstm_93/while/lstm_cell/mulMul.backward_lstm_93/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
%backward_lstm_93/while/lstm_cell/ReluRelu/backward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/mul_1Mul,backward_lstm_93/while/lstm_cell/Sigmoid:y:03backward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/add_1AddV2(backward_lstm_93/while/lstm_cell/mul:z:0*backward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
*backward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
'backward_lstm_93/while/lstm_cell/Relu_1Relu*backward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/mul_2Mul.backward_lstm_93/while/lstm_cell/Sigmoid_2:y:05backward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
;backward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_93_while_placeholder_1"backward_lstm_93_while_placeholder*backward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���^
backward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/addAddV2"backward_lstm_93_while_placeholder%backward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/add_1AddV2:backward_lstm_93_while_backward_lstm_93_while_loop_counter'backward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
backward_lstm_93/while/IdentityIdentity backward_lstm_93/while/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_1Identity@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_2Identitybackward_lstm_93/while/add:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_3IdentityKbackward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_4Identity*backward_lstm_93/while/lstm_cell/mul_2:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
!backward_lstm_93/while/Identity_5Identity*backward_lstm_93/while/lstm_cell/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
backward_lstm_93/while/NoOpNoOp8^backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_93_while_backward_lstm_93_strided_slice_19backward_lstm_93_while_backward_lstm_93_strided_slice_1_0"K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0"O
!backward_lstm_93_while_identity_1*backward_lstm_93/while/Identity_1:output:0"O
!backward_lstm_93_while_identity_2*backward_lstm_93/while/Identity_2:output:0"O
!backward_lstm_93_while_identity_3*backward_lstm_93/while/Identity_3:output:0"O
!backward_lstm_93_while_identity_4*backward_lstm_93/while/Identity_4:output:0"O
!backward_lstm_93_while_identity_5*backward_lstm_93/while/Identity_5:output:0"�
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2r
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�J
�
#forward_lstm_93_while_body_55913471<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3;
7forward_lstm_93_while_forward_lstm_93_strided_slice_1_0w
sforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_93_while_identity$
 forward_lstm_93_while_identity_1$
 forward_lstm_93_while_identity_2$
 forward_lstm_93_while_identity_3$
 forward_lstm_93_while_identity_4$
 forward_lstm_93_while_identity_59
5forward_lstm_93_while_forward_lstm_93_strided_slice_1u
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Gforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
9forward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_93_while_placeholderPforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
&forward_lstm_93/while/lstm_cell/MatMulMatMul@forward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
(forward_lstm_93/while/lstm_cell/MatMul_1MatMul#forward_lstm_93_while_placeholder_2?forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
#forward_lstm_93/while/lstm_cell/addAddV20forward_lstm_93/while/lstm_cell/MatMul:product:02forward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
'forward_lstm_93/while/lstm_cell/BiasAddBiasAdd'forward_lstm_93/while/lstm_cell/add:z:0>forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
/forward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
%forward_lstm_93/while/lstm_cell/splitSplit8forward_lstm_93/while/lstm_cell/split/split_dim:output:00forward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
'forward_lstm_93/while/lstm_cell/SigmoidSigmoid.forward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
)forward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
#forward_lstm_93/while/lstm_cell/mulMul-forward_lstm_93/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
$forward_lstm_93/while/lstm_cell/ReluRelu.forward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/mul_1Mul+forward_lstm_93/while/lstm_cell/Sigmoid:y:02forward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/add_1AddV2'forward_lstm_93/while/lstm_cell/mul:z:0)forward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
)forward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
&forward_lstm_93/while/lstm_cell/Relu_1Relu)forward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/mul_2Mul-forward_lstm_93/while/lstm_cell/Sigmoid_2:y:04forward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
:forward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_93_while_placeholder_1!forward_lstm_93_while_placeholder)forward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���]
forward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/addAddV2!forward_lstm_93_while_placeholder$forward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/add_1AddV28forward_lstm_93_while_forward_lstm_93_while_loop_counter&forward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_1Identity>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_2Identityforward_lstm_93/while/add:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_3IdentityJforward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_4Identity)forward_lstm_93/while/lstm_cell/mul_2:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
 forward_lstm_93/while/Identity_5Identity)forward_lstm_93/while/lstm_cell/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
forward_lstm_93/while/NoOpNoOp7^forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_93_while_forward_lstm_93_strided_slice_17forward_lstm_93_while_forward_lstm_93_strided_slice_1_0"I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0"M
 forward_lstm_93_while_identity_1)forward_lstm_93/while/Identity_1:output:0"M
 forward_lstm_93_while_identity_2)forward_lstm_93/while/Identity_2:output:0"M
 forward_lstm_93_while_identity_3)forward_lstm_93/while/Identity_3:output:0"M
 forward_lstm_93_while_identity_4)forward_lstm_93/while/Identity_4:output:0"M
 forward_lstm_93_while_identity_5)forward_lstm_93/while/Identity_5:output:0"�
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2p
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�K
�
$backward_lstm_93_while_body_55912042>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3=
9backward_lstm_93_while_backward_lstm_93_strided_slice_1_0y
ubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_93_while_identity%
!backward_lstm_93_while_identity_1%
!backward_lstm_93_while_identity_2%
!backward_lstm_93_while_identity_3%
!backward_lstm_93_while_identity_4%
!backward_lstm_93_while_identity_5;
7backward_lstm_93_while_backward_lstm_93_strided_slice_1w
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Hbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
:backward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_93_while_placeholderQbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
'backward_lstm_93/while/lstm_cell/MatMulMatMulAbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
)backward_lstm_93/while/lstm_cell/MatMul_1MatMul$backward_lstm_93_while_placeholder_2@backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
$backward_lstm_93/while/lstm_cell/addAddV21backward_lstm_93/while/lstm_cell/MatMul:product:03backward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
(backward_lstm_93/while/lstm_cell/BiasAddBiasAdd(backward_lstm_93/while/lstm_cell/add:z:0?backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: r
0backward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
&backward_lstm_93/while/lstm_cell/splitSplit9backward_lstm_93/while/lstm_cell/split/split_dim:output:01backward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
(backward_lstm_93/while/lstm_cell/SigmoidSigmoid/backward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
*backward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
$backward_lstm_93/while/lstm_cell/mulMul.backward_lstm_93/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
%backward_lstm_93/while/lstm_cell/ReluRelu/backward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/mul_1Mul,backward_lstm_93/while/lstm_cell/Sigmoid:y:03backward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/add_1AddV2(backward_lstm_93/while/lstm_cell/mul:z:0*backward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
*backward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
'backward_lstm_93/while/lstm_cell/Relu_1Relu*backward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/mul_2Mul.backward_lstm_93/while/lstm_cell/Sigmoid_2:y:05backward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
;backward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_93_while_placeholder_1"backward_lstm_93_while_placeholder*backward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���^
backward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/addAddV2"backward_lstm_93_while_placeholder%backward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/add_1AddV2:backward_lstm_93_while_backward_lstm_93_while_loop_counter'backward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
backward_lstm_93/while/IdentityIdentity backward_lstm_93/while/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_1Identity@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_2Identitybackward_lstm_93/while/add:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_3IdentityKbackward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_4Identity*backward_lstm_93/while/lstm_cell/mul_2:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
!backward_lstm_93/while/Identity_5Identity*backward_lstm_93/while/lstm_cell/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
backward_lstm_93/while/NoOpNoOp8^backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_93_while_backward_lstm_93_strided_slice_19backward_lstm_93_while_backward_lstm_93_strided_slice_1_0"K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0"O
!backward_lstm_93_while_identity_1*backward_lstm_93/while/Identity_1:output:0"O
!backward_lstm_93_while_identity_2*backward_lstm_93/while/Identity_2:output:0"O
!backward_lstm_93_while_identity_3*backward_lstm_93/while/Identity_3:output:0"O
!backward_lstm_93_while_identity_4*backward_lstm_93/while/Identity_4:output:0"O
!backward_lstm_93_while_identity_5*backward_lstm_93/while/Identity_5:output:0"�
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2r
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�8
�
while_body_55914163
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
$backward_lstm_93_while_cond_55913611>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3@
<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913611___redundant_placeholder0X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913611___redundant_placeholder1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913611___redundant_placeholder2X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913611___redundant_placeholder3#
backward_lstm_93_while_identity
�
backward_lstm_93/while/LessLess"backward_lstm_93_while_placeholder<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_93/while/IdentityIdentitybackward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:

_output_shapes
:
�J
�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914533

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55914449*
condR
while_cond_55914448*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
while_cond_55910195
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55910195___redundant_placeholder06
2while_while_cond_55910195___redundant_placeholder16
2while_while_cond_55910195___redundant_placeholder26
2while_while_cond_55910195___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
��
�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55912130

inputsJ
8forward_lstm_93_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_93_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource: 
identity��1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp�2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�backward_lstm_93/while�0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp�1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�forward_lstm_93/whilej
forward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         m
#forward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_sliceStridedSliceforward_lstm_93/Shape:output:0,forward_lstm_93/strided_slice/stack:output:0.forward_lstm_93/strided_slice/stack_1:output:0.forward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros/packedPack&forward_lstm_93/strided_slice:output:0'forward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zerosFill%forward_lstm_93/zeros/packed:output:0$forward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:b
 forward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros_1/packedPack&forward_lstm_93/strided_slice:output:0)forward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zeros_1Fill'forward_lstm_93/zeros_1/packed:output:0&forward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:s
forward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose	Transposeinputs'forward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:l
forward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         o
%forward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_1StridedSlice forward_lstm_93/Shape_1:output:0.forward_lstm_93/strided_slice_1/stack:output:00forward_lstm_93/strided_slice_1/stack_1:output:00forward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
forward_lstm_93/TensorArrayV2TensorListReserve4forward_lstm_93/TensorArrayV2/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Eforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
7forward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_93/transpose:y:0Nforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���o
%forward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_2StridedSliceforward_lstm_93/transpose:y:0.forward_lstm_93/strided_slice_2/stack:output:00forward_lstm_93/strided_slice_2/stack_1:output:00forward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 forward_lstm_93/lstm_cell/MatMulMatMul(forward_lstm_93/strided_slice_2:output:07forward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
"forward_lstm_93/lstm_cell/MatMul_1MatMulforward_lstm_93/zeros:output:09forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
forward_lstm_93/lstm_cell/addAddV2*forward_lstm_93/lstm_cell/MatMul:product:0,forward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!forward_lstm_93/lstm_cell/BiasAddBiasAdd!forward_lstm_93/lstm_cell/add:z:08forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: k
)forward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/lstm_cell/splitSplit2forward_lstm_93/lstm_cell/split/split_dim:output:0*forward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split
!forward_lstm_93/lstm_cell/SigmoidSigmoid(forward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
#forward_lstm_93/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mulMul'forward_lstm_93/lstm_cell/Sigmoid_1:y:0 forward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:y
forward_lstm_93/lstm_cell/ReluRelu(forward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mul_1Mul%forward_lstm_93/lstm_cell/Sigmoid:y:0,forward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/add_1AddV2!forward_lstm_93/lstm_cell/mul:z:0#forward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
#forward_lstm_93/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:v
 forward_lstm_93/lstm_cell/Relu_1Relu#forward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mul_2Mul'forward_lstm_93/lstm_cell/Sigmoid_2:y:0.forward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:~
-forward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
forward_lstm_93/TensorArrayV2_1TensorListReserve6forward_lstm_93/TensorArrayV2_1/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���V
forward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������d
"forward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
forward_lstm_93/whileWhile+forward_lstm_93/while/loop_counter:output:01forward_lstm_93/while/maximum_iterations:output:0forward_lstm_93/time:output:0(forward_lstm_93/TensorArrayV2_1:handle:0forward_lstm_93/zeros:output:0 forward_lstm_93/zeros_1:output:0(forward_lstm_93/strided_slice_1:output:0Gforward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_93_lstm_cell_matmul_readvariableop_resource:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#forward_lstm_93_while_body_55911901*/
cond'R%
#forward_lstm_93_while_cond_55911900*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
@forward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
2forward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_93/while:output:3Iforward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0x
%forward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������q
'forward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_3StridedSlice;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_93/strided_slice_3/stack:output:00forward_lstm_93/strided_slice_3/stack_1:output:00forward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masku
 forward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose_1	Transpose;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:k
forward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
backward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         n
$backward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
backward_lstm_93/strided_sliceStridedSlicebackward_lstm_93/Shape:output:0-backward_lstm_93/strided_slice/stack:output:0/backward_lstm_93/strided_slice/stack_1:output:0/backward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros/packedPack'backward_lstm_93/strided_slice:output:0(backward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zerosFill&backward_lstm_93/zeros/packed:output:0%backward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:c
!backward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros_1/packedPack'backward_lstm_93/strided_slice:output:0*backward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zeros_1Fill(backward_lstm_93/zeros_1/packed:output:0'backward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:t
backward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose	Transposeinputs(backward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:m
backward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
&backward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_1StridedSlice!backward_lstm_93/Shape_1:output:0/backward_lstm_93/strided_slice_1/stack:output:01backward_lstm_93/strided_slice_1/stack_1:output:01backward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
backward_lstm_93/TensorArrayV2TensorListReserve5backward_lstm_93/TensorArrayV2/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���i
backward_lstm_93/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
backward_lstm_93/ReverseV2	ReverseV2backward_lstm_93/transpose:y:0(backward_lstm_93/ReverseV2/axis:output:0*
T0*"
_output_shapes
:�
Fbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
8backward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_93/ReverseV2:output:0Obackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���p
&backward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_2StridedSlicebackward_lstm_93/transpose:y:0/backward_lstm_93/strided_slice_2/stack:output:01backward_lstm_93/strided_slice_2/stack_1:output:01backward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
!backward_lstm_93/lstm_cell/MatMulMatMul)backward_lstm_93/strided_slice_2:output:08backward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
#backward_lstm_93/lstm_cell/MatMul_1MatMulbackward_lstm_93/zeros:output:0:backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
backward_lstm_93/lstm_cell/addAddV2+backward_lstm_93/lstm_cell/MatMul:product:0-backward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"backward_lstm_93/lstm_cell/BiasAddBiasAdd"backward_lstm_93/lstm_cell/add:z:09backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: l
*backward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 backward_lstm_93/lstm_cell/splitSplit3backward_lstm_93/lstm_cell/split/split_dim:output:0+backward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
"backward_lstm_93/lstm_cell/SigmoidSigmoid)backward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
$backward_lstm_93/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
backward_lstm_93/lstm_cell/mulMul(backward_lstm_93/lstm_cell/Sigmoid_1:y:0!backward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:{
backward_lstm_93/lstm_cell/ReluRelu)backward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/mul_1Mul&backward_lstm_93/lstm_cell/Sigmoid:y:0-backward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/add_1AddV2"backward_lstm_93/lstm_cell/mul:z:0$backward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
$backward_lstm_93/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:x
!backward_lstm_93/lstm_cell/Relu_1Relu$backward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/mul_2Mul(backward_lstm_93/lstm_cell/Sigmoid_2:y:0/backward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:
.backward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 backward_lstm_93/TensorArrayV2_1TensorListReserve7backward_lstm_93/TensorArrayV2_1/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���W
backward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������e
#backward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
backward_lstm_93/whileWhile,backward_lstm_93/while/loop_counter:output:02backward_lstm_93/while/maximum_iterations:output:0backward_lstm_93/time:output:0)backward_lstm_93/TensorArrayV2_1:handle:0backward_lstm_93/zeros:output:0!backward_lstm_93/zeros_1:output:0)backward_lstm_93/strided_slice_1:output:0Hbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_93_lstm_cell_matmul_readvariableop_resource;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$backward_lstm_93_while_body_55912042*0
cond(R&
$backward_lstm_93_while_cond_55912041*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
Abackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
3backward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_93/while:output:3Jbackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0y
&backward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������r
(backward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_3StridedSlice<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_93/strided_slice_3/stack:output:01backward_lstm_93/strided_slice_3/stack_1:output:01backward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskv
!backward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose_1	Transpose<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:l
backward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:~
	ReverseV2	ReverseV2 backward_lstm_93/transpose_1:y:0ReverseV2/axis:output:0*
T0*"
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2forward_lstm_93/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*"
_output_shapes
:Y
IdentityIdentityconcat:output:0^NoOp*
T0*"
_output_shapes
:�
NoOpNoOp2^backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_93/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_93/while1^forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_93/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2f
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_93/whilebackward_lstm_93/while2d
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_93/whileforward_lstm_93/while:J F
"
_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�8
�
while_body_55915073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�L
�
#forward_lstm_93_while_body_55912895<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3;
7forward_lstm_93_while_forward_lstm_93_strided_slice_1_0w
sforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_93_while_identity$
 forward_lstm_93_while_identity_1$
 forward_lstm_93_while_identity_2$
 forward_lstm_93_while_identity_3$
 forward_lstm_93_while_identity_4$
 forward_lstm_93_while_identity_59
5forward_lstm_93_while_forward_lstm_93_strided_slice_1u
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Gforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
9forward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_93_while_placeholderPforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
&forward_lstm_93/while/lstm_cell/MatMulMatMul@forward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
(forward_lstm_93/while/lstm_cell/MatMul_1MatMul#forward_lstm_93_while_placeholder_2?forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
#forward_lstm_93/while/lstm_cell/addAddV20forward_lstm_93/while/lstm_cell/MatMul:product:02forward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
'forward_lstm_93/while/lstm_cell/BiasAddBiasAdd'forward_lstm_93/while/lstm_cell/add:z:0>forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� q
/forward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
%forward_lstm_93/while/lstm_cell/splitSplit8forward_lstm_93/while/lstm_cell/split/split_dim:output:00forward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
'forward_lstm_93/while/lstm_cell/SigmoidSigmoid.forward_lstm_93/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:����������
)forward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_93/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
#forward_lstm_93/while/lstm_cell/mulMul-forward_lstm_93/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_93_while_placeholder_3*
T0*'
_output_shapes
:����������
$forward_lstm_93/while/lstm_cell/ReluRelu.forward_lstm_93/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
%forward_lstm_93/while/lstm_cell/mul_1Mul+forward_lstm_93/while/lstm_cell/Sigmoid:y:02forward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
%forward_lstm_93/while/lstm_cell/add_1AddV2'forward_lstm_93/while/lstm_cell/mul:z:0)forward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:����������
)forward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_93/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:����������
&forward_lstm_93/while/lstm_cell/Relu_1Relu)forward_lstm_93/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
%forward_lstm_93/while/lstm_cell/mul_2Mul-forward_lstm_93/while/lstm_cell/Sigmoid_2:y:04forward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
:forward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_93_while_placeholder_1!forward_lstm_93_while_placeholder)forward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���]
forward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/addAddV2!forward_lstm_93_while_placeholder$forward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/add_1AddV28forward_lstm_93_while_forward_lstm_93_while_loop_counter&forward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_1Identity>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_2Identityforward_lstm_93/while/add:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_3IdentityJforward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_4Identity)forward_lstm_93/while/lstm_cell/mul_2:z:0^forward_lstm_93/while/NoOp*
T0*'
_output_shapes
:����������
 forward_lstm_93/while/Identity_5Identity)forward_lstm_93/while/lstm_cell/add_1:z:0^forward_lstm_93/while/NoOp*
T0*'
_output_shapes
:����������
forward_lstm_93/while/NoOpNoOp7^forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_93_while_forward_lstm_93_strided_slice_17forward_lstm_93_while_forward_lstm_93_strided_slice_1_0"I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0"M
 forward_lstm_93_while_identity_1)forward_lstm_93/while/Identity_1:output:0"M
 forward_lstm_93_while_identity_2)forward_lstm_93/while/Identity_2:output:0"M
 forward_lstm_93_while_identity_3)forward_lstm_93/while/Identity_3:output:0"M
 forward_lstm_93_while_identity_4)forward_lstm_93/while/Identity_4:output:0"M
 forward_lstm_93_while_identity_5)forward_lstm_93/while/Identity_5:output:0"�
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2p
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
$backward_lstm_93_while_cond_55911643>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3@
<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55911643___redundant_placeholder0X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55911643___redundant_placeholder1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55911643___redundant_placeholder2X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55911643___redundant_placeholder3#
backward_lstm_93_while_identity
�
backward_lstm_93/while/LessLess"backward_lstm_93_while_placeholder<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_93/while/IdentityIdentitybackward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:

_output_shapes
:
�	
�
3__inference_backward_lstm_93_layer_call_fn_55914555
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55910615|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55914547:($
"
_user_specified_name
55914549:($
"
_user_specified_name
55914551
�	
�
while_cond_55914162
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55914162___redundant_placeholder06
2while_while_cond_55914162___redundant_placeholder16
2while_while_cond_55914162___redundant_placeholder26
2while_while_cond_55914162___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
Y
-__inference_lambda_100_layer_call_fn_55912444
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lambda_100_layer_call_and_return_conditional_losses_55911809_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:::L H
"
_output_shapes
:
"
_user_specified_name
inputs_0:LH
"
_output_shapes
:
"
_user_specified_name
inputs_1
�
t
H__inference_lambda_100_layer_call_and_return_conditional_losses_55912450
inputs_0
inputs_1
identity�
einsum/EinsumEinsuminputs_0inputs_1*
N*
T0*&
_output_shapes
:*
equationabc,abd->dabc]
IdentityIdentityeinsum/Einsum:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:::L H
"
_output_shapes
:
"
_user_specified_name
inputs_0:LH
"
_output_shapes
:
"
_user_specified_name
inputs_1
�
Y
-__inference_lambda_101_layer_call_fn_55912462
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lambda_101_layer_call_and_return_conditional_losses_55911443[
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::P L
&
_output_shapes
:
"
_user_specified_name
inputs_0:LH
"
_output_shapes
:
"
_user_specified_name
inputs_1
�J
�
#forward_lstm_93_while_body_55911503<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3;
7forward_lstm_93_while_forward_lstm_93_strided_slice_1_0w
sforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_93_while_identity$
 forward_lstm_93_while_identity_1$
 forward_lstm_93_while_identity_2$
 forward_lstm_93_while_identity_3$
 forward_lstm_93_while_identity_4$
 forward_lstm_93_while_identity_59
5forward_lstm_93_while_forward_lstm_93_strided_slice_1u
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Gforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
9forward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_93_while_placeholderPforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
&forward_lstm_93/while/lstm_cell/MatMulMatMul@forward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
(forward_lstm_93/while/lstm_cell/MatMul_1MatMul#forward_lstm_93_while_placeholder_2?forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
#forward_lstm_93/while/lstm_cell/addAddV20forward_lstm_93/while/lstm_cell/MatMul:product:02forward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
'forward_lstm_93/while/lstm_cell/BiasAddBiasAdd'forward_lstm_93/while/lstm_cell/add:z:0>forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
/forward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
%forward_lstm_93/while/lstm_cell/splitSplit8forward_lstm_93/while/lstm_cell/split/split_dim:output:00forward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
'forward_lstm_93/while/lstm_cell/SigmoidSigmoid.forward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
)forward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
#forward_lstm_93/while/lstm_cell/mulMul-forward_lstm_93/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
$forward_lstm_93/while/lstm_cell/ReluRelu.forward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/mul_1Mul+forward_lstm_93/while/lstm_cell/Sigmoid:y:02forward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/add_1AddV2'forward_lstm_93/while/lstm_cell/mul:z:0)forward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
)forward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
&forward_lstm_93/while/lstm_cell/Relu_1Relu)forward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/mul_2Mul-forward_lstm_93/while/lstm_cell/Sigmoid_2:y:04forward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
:forward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_93_while_placeholder_1!forward_lstm_93_while_placeholder)forward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���]
forward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/addAddV2!forward_lstm_93_while_placeholder$forward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/add_1AddV28forward_lstm_93_while_forward_lstm_93_while_loop_counter&forward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_1Identity>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_2Identityforward_lstm_93/while/add:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_3IdentityJforward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_4Identity)forward_lstm_93/while/lstm_cell/mul_2:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
 forward_lstm_93/while/Identity_5Identity)forward_lstm_93/while/lstm_cell/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
forward_lstm_93/while/NoOpNoOp7^forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_93_while_forward_lstm_93_strided_slice_17forward_lstm_93_while_forward_lstm_93_strided_slice_1_0"I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0"M
 forward_lstm_93_while_identity_1)forward_lstm_93/while/Identity_1:output:0"M
 forward_lstm_93_while_identity_2)forward_lstm_93/while/Identity_2:output:0"M
 forward_lstm_93_while_identity_3)forward_lstm_93/while/Identity_3:output:0"M
 forward_lstm_93_while_identity_4)forward_lstm_93/while/Identity_4:output:0"M
 forward_lstm_93_while_identity_5)forward_lstm_93/while/Identity_5:output:0"�
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2p
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_cond_55909432~
ztopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_loop_counter�
�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_maximum_iterationsF
Btopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholderH
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder_1H
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder_2H
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder_3�
|topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_less_topk_bilstm_moe_bidirectional_93_forward_lstm_93_strided_slice_1�
�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_cond_55909432___redundant_placeholder0�
�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_cond_55909432___redundant_placeholder1�
�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_cond_55909432___redundant_placeholder2�
�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_cond_55909432___redundant_placeholder3C
?topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity
�
;topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/LessLessBtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder|topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_less_topk_bilstm_moe_bidirectional_93_forward_lstm_93_strided_slice_1*
T0*
_output_shapes
: �
?topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/IdentityIdentity?topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "�
?topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identityHtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::{ w

_output_shapes
: 
]
_user_specified_nameECtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/loop_counter:�}

_output_shapes
: 
c
_user_specified_nameKItopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::xt

_output_shapes
: 
Z
_user_specified_nameB@topk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1:

_output_shapes
:
�
r
H__inference_lambda_100_layer_call_and_return_conditional_losses_55911411

inputs
inputs_1
identity�
einsum/EinsumEinsuminputsinputs_1*
N*
T0*&
_output_shapes
:*
equationabc,abd->dabc]
IdentityIdentityeinsum/Einsum:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:::J F
"
_output_shapes
:
 
_user_specified_nameinputs:JF
"
_output_shapes
:
 
_user_specified_nameinputs
�J
�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55911126

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55911042*
condR
while_cond_55911041*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�8
�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55910265

inputs$
lstm_cell_55910183: $
lstm_cell_55910185:  
lstm_cell_55910187: 
identity��!lstm_cell/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55910183lstm_cell_55910185lstm_cell_55910187*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910182n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55910183lstm_cell_55910185lstm_cell_55910187*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55910196*
condR
while_cond_55910195*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55910183:($
"
_user_specified_name
55910185:($
"
_user_specified_name
55910187
�
�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55911291

inputs*
forward_lstm_93_55911127: *
forward_lstm_93_55911129: &
forward_lstm_93_55911131: +
backward_lstm_93_55911279: +
backward_lstm_93_55911281: '
backward_lstm_93_55911283: 
identity��(backward_lstm_93/StatefulPartitionedCall�'forward_lstm_93/StatefulPartitionedCall�
'forward_lstm_93/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_93_55911127forward_lstm_93_55911129forward_lstm_93_55911131*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55911126�
(backward_lstm_93/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_93_55911279backward_lstm_93_55911281backward_lstm_93_55911283*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55911278X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
	ReverseV2	ReverseV21backward_lstm_93/StatefulPartitionedCall:output:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV20forward_lstm_93/StatefulPartitionedCall:output:0ReverseV2:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :������������������k
IdentityIdentityconcat:output:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp)^backward_lstm_93/StatefulPartitionedCall(^forward_lstm_93/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2T
(backward_lstm_93/StatefulPartitionedCall(backward_lstm_93/StatefulPartitionedCall2R
'forward_lstm_93/StatefulPartitionedCall'forward_lstm_93/StatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55911127:($
"
_user_specified_name
55911129:($
"
_user_specified_name
55911131:($
"
_user_specified_name
55911279:($
"
_user_specified_name
55911281:($
"
_user_specified_name
55911283
�
d
H__inference_flatten_93_layer_call_and_return_conditional_losses_55913738

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  T
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	�P
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�:
�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55910615

inputs$
lstm_cell_55910533: $
lstm_cell_55910535:  
lstm_cell_55910537: 
identity��!lstm_cell/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :�������������������
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55910533lstm_cell_55910535lstm_cell_55910537*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910532n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55910533lstm_cell_55910535lstm_cell_55910537*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55910546*
condR
while_cond_55910545*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55910533:($
"
_user_specified_name
55910535:($
"
_user_specified_name
55910537
�	
�
2__inference_forward_lstm_93_layer_call_fn_55913961

inputs
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55911126|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55913953:($
"
_user_specified_name
55913955:($
"
_user_specified_name
55913957
�
�
&__inference_signature_wrapper_55912384
input_4
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15:	�

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_55909671f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:
!
_user_specified_name	input_4:($
"
_user_specified_name
55912346:($
"
_user_specified_name
55912348:($
"
_user_specified_name
55912350:($
"
_user_specified_name
55912352:($
"
_user_specified_name
55912354:($
"
_user_specified_name
55912356:($
"
_user_specified_name
55912358:($
"
_user_specified_name
55912360:(	$
"
_user_specified_name
55912362:(
$
"
_user_specified_name
55912364:($
"
_user_specified_name
55912366:($
"
_user_specified_name
55912368:($
"
_user_specified_name
55912370:($
"
_user_specified_name
55912372:($
"
_user_specified_name
55912374:($
"
_user_specified_name
55912376:($
"
_user_specified_name
55912378:($
"
_user_specified_name
55912380
�8
�
while_body_55914449
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�	
�
while_cond_55914448
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55914448___redundant_placeholder06
2while_while_cond_55914448___redundant_placeholder16
2while_while_cond_55914448___redundant_placeholder26
2while_while_cond_55914448___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
�
,__inference_dense_605_layer_call_fn_55913846

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_605_layer_call_and_return_conditional_losses_55909856s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
55913840:($
"
_user_specified_name
55913842
�
I
-__inference_flatten_93_layer_call_fn_55913732

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_flatten_93_layer_call_and_return_conditional_losses_55911764X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�	
�
3__inference_backward_lstm_93_layer_call_fn_55914544
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55910468|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55914536:($
"
_user_specified_name
55914538:($
"
_user_specified_name
55914540
�J
�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914104
inputs_0:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55914020*
condR
while_cond_55914019*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
#forward_lstm_93_while_cond_55911900<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3>
:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55911900___redundant_placeholder0V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55911900___redundant_placeholder1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55911900___redundant_placeholder2V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55911900___redundant_placeholder3"
forward_lstm_93_while_identity
�
forward_lstm_93/while/LessLess!forward_lstm_93_while_placeholder:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:

_output_shapes
:
�K
�
$backward_lstm_93_while_body_55913612>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3=
9backward_lstm_93_while_backward_lstm_93_strided_slice_1_0y
ubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_93_while_identity%
!backward_lstm_93_while_identity_1%
!backward_lstm_93_while_identity_2%
!backward_lstm_93_while_identity_3%
!backward_lstm_93_while_identity_4%
!backward_lstm_93_while_identity_5;
7backward_lstm_93_while_backward_lstm_93_strided_slice_1w
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Hbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
:backward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_93_while_placeholderQbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
'backward_lstm_93/while/lstm_cell/MatMulMatMulAbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
)backward_lstm_93/while/lstm_cell/MatMul_1MatMul$backward_lstm_93_while_placeholder_2@backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
$backward_lstm_93/while/lstm_cell/addAddV21backward_lstm_93/while/lstm_cell/MatMul:product:03backward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
(backward_lstm_93/while/lstm_cell/BiasAddBiasAdd(backward_lstm_93/while/lstm_cell/add:z:0?backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: r
0backward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
&backward_lstm_93/while/lstm_cell/splitSplit9backward_lstm_93/while/lstm_cell/split/split_dim:output:01backward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
(backward_lstm_93/while/lstm_cell/SigmoidSigmoid/backward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
*backward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
$backward_lstm_93/while/lstm_cell/mulMul.backward_lstm_93/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
%backward_lstm_93/while/lstm_cell/ReluRelu/backward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/mul_1Mul,backward_lstm_93/while/lstm_cell/Sigmoid:y:03backward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/add_1AddV2(backward_lstm_93/while/lstm_cell/mul:z:0*backward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
*backward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
'backward_lstm_93/while/lstm_cell/Relu_1Relu*backward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
&backward_lstm_93/while/lstm_cell/mul_2Mul.backward_lstm_93/while/lstm_cell/Sigmoid_2:y:05backward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
;backward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_93_while_placeholder_1"backward_lstm_93_while_placeholder*backward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���^
backward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/addAddV2"backward_lstm_93_while_placeholder%backward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/add_1AddV2:backward_lstm_93_while_backward_lstm_93_while_loop_counter'backward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
backward_lstm_93/while/IdentityIdentity backward_lstm_93/while/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_1Identity@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_2Identitybackward_lstm_93/while/add:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_3IdentityKbackward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_4Identity*backward_lstm_93/while/lstm_cell/mul_2:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
!backward_lstm_93/while/Identity_5Identity*backward_lstm_93/while/lstm_cell/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
backward_lstm_93/while/NoOpNoOp8^backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_93_while_backward_lstm_93_strided_slice_19backward_lstm_93_while_backward_lstm_93_strided_slice_1_0"K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0"O
!backward_lstm_93_while_identity_1*backward_lstm_93/while/Identity_1:output:0"O
!backward_lstm_93_while_identity_2*backward_lstm_93/while/Identity_2:output:0"O
!backward_lstm_93_while_identity_3*backward_lstm_93/while/Identity_3:output:0"O
!backward_lstm_93_while_identity_4*backward_lstm_93/while/Identity_4:output:0"O
!backward_lstm_93_while_identity_5*backward_lstm_93/while/Identity_5:output:0"�
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2r
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�H
�	
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55911782
input_4$
dense_594_55911387: 
dense_594_55911389:)
sequential_408_55911417:%
sequential_408_55911419:)
sequential_413_55911422:%
sequential_413_55911424:)
sequential_418_55911427:%
sequential_418_55911429:)
sequential_423_55911432:%
sequential_423_55911434:+
bidirectional_93_55911733: +
bidirectional_93_55911735: '
bidirectional_93_55911737: +
bidirectional_93_55911739: +
bidirectional_93_55911741: '
bidirectional_93_55911743: %
dense_611_55911776:	� 
dense_611_55911778:
identity��(bidirectional_93/StatefulPartitionedCall�!dense_594/StatefulPartitionedCall�!dense_611/StatefulPartitionedCall�"dropout_93/StatefulPartitionedCall�&sequential_408/StatefulPartitionedCall�&sequential_413/StatefulPartitionedCall�&sequential_418/StatefulPartitionedCall�&sequential_423/StatefulPartitionedCall�
!dense_594/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_594_55911387dense_594_55911389*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_594_layer_call_and_return_conditional_losses_55911386Z
tf.math.top_k_3/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.top_k_3/TopKV2TopKV2*dense_594/StatefulPartitionedCall:output:0!tf.math.top_k_3/TopKV2/k:output:0*
T0*0
_output_shapes
::b
tf.one_hot_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
tf.one_hot_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    \
tf.one_hot_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
tf.one_hot_3/one_hotOneHot tf.math.top_k_3/TopKV2:indices:0#tf.one_hot_3/one_hot/depth:output:0&tf.one_hot_3/one_hot/on_value:output:0'tf.one_hot_3/one_hot/off_value:output:0*
T0*
TI0*&
_output_shapes
:�
lambda_99/PartitionedCallPartitionedCalltf.math.top_k_3/TopKV2:values:0tf.one_hot_3/one_hot:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lambda_99_layer_call_and_return_conditional_losses_55911404�
lambda_100/PartitionedCallPartitionedCallinput_4"lambda_99/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lambda_100_layer_call_and_return_conditional_losses_55911411�
tf.unstack_3/unstackUnpack#lambda_100/PartitionedCall:output:0*
T0*L
_output_shapes:
8::::*	
num�
&sequential_408/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_3/unstack:output:0sequential_408_55911417sequential_408_55911419*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909711�
&sequential_413/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_3/unstack:output:1sequential_413_55911422sequential_413_55911424*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909787�
&sequential_418/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_3/unstack:output:2sequential_418_55911427sequential_418_55911429*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909863�
&sequential_423/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_3/unstack:output:3sequential_423_55911432sequential_423_55911434*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909939�
tf.stack_94/stackPack/sequential_408/StatefulPartitionedCall:output:0/sequential_413/StatefulPartitionedCall:output:0/sequential_418/StatefulPartitionedCall:output:0/sequential_423/StatefulPartitionedCall:output:0*
N*
T0*&
_output_shapes
:*

axis�
lambda_101/PartitionedCallPartitionedCalltf.stack_94/stack:output:0"lambda_99/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lambda_101_layer_call_and_return_conditional_losses_55911443�
(bidirectional_93/StatefulPartitionedCallStatefulPartitionedCall#lambda_101/PartitionedCall:output:0bidirectional_93_55911733bidirectional_93_55911735bidirectional_93_55911737bidirectional_93_55911739bidirectional_93_55911741bidirectional_93_55911743*
Tin
	2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55911732�
"dropout_93/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_93/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_93_layer_call_and_return_conditional_losses_55911757�
flatten_93/PartitionedCallPartitionedCall+dropout_93/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_flatten_93_layer_call_and_return_conditional_losses_55911764�
!dense_611/StatefulPartitionedCallStatefulPartitionedCall#flatten_93/PartitionedCall:output:0dense_611_55911776dense_611_55911778*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_611_layer_call_and_return_conditional_losses_55911775p
IdentityIdentity*dense_611/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp)^bidirectional_93/StatefulPartitionedCall"^dense_594/StatefulPartitionedCall"^dense_611/StatefulPartitionedCall#^dropout_93/StatefulPartitionedCall'^sequential_408/StatefulPartitionedCall'^sequential_413/StatefulPartitionedCall'^sequential_418/StatefulPartitionedCall'^sequential_423/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 2T
(bidirectional_93/StatefulPartitionedCall(bidirectional_93/StatefulPartitionedCall2F
!dense_594/StatefulPartitionedCall!dense_594/StatefulPartitionedCall2F
!dense_611/StatefulPartitionedCall!dense_611/StatefulPartitionedCall2H
"dropout_93/StatefulPartitionedCall"dropout_93/StatefulPartitionedCall2P
&sequential_408/StatefulPartitionedCall&sequential_408/StatefulPartitionedCall2P
&sequential_413/StatefulPartitionedCall&sequential_413/StatefulPartitionedCall2P
&sequential_418/StatefulPartitionedCall&sequential_418/StatefulPartitionedCall2P
&sequential_423/StatefulPartitionedCall&sequential_423/StatefulPartitionedCall:K G
"
_output_shapes
:
!
_user_specified_name	input_4:($
"
_user_specified_name
55911387:($
"
_user_specified_name
55911389:($
"
_user_specified_name
55911417:($
"
_user_specified_name
55911419:($
"
_user_specified_name
55911422:($
"
_user_specified_name
55911424:($
"
_user_specified_name
55911427:($
"
_user_specified_name
55911429:(	$
"
_user_specified_name
55911432:(
$
"
_user_specified_name
55911434:($
"
_user_specified_name
55911733:($
"
_user_specified_name
55911735:($
"
_user_specified_name
55911737:($
"
_user_specified_name
55911739:($
"
_user_specified_name
55911741:($
"
_user_specified_name
55911743:($
"
_user_specified_name
55911776:($
"
_user_specified_name
55911778
�8
�
while_body_55914020
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
I
-__inference_dropout_93_layer_call_fn_55913710

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_93_layer_call_and_return_conditional_losses_55912148[
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
G__inference_dense_595_layer_call_and_return_conditional_losses_55909704

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
$backward_lstm_93_while_cond_55912747>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3@
<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55912747___redundant_placeholder0X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55912747___redundant_placeholder1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55912747___redundant_placeholder2X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55912747___redundant_placeholder3#
backward_lstm_93_while_identity
�
backward_lstm_93/while/LessLess"backward_lstm_93_while_placeholder<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_93/while/IdentityIdentitybackward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:

_output_shapes
:
��
�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913412

inputsJ
8forward_lstm_93_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_93_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource: 
identity��1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp�2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�backward_lstm_93/while�0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp�1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�forward_lstm_93/whilej
forward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         m
#forward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_sliceStridedSliceforward_lstm_93/Shape:output:0,forward_lstm_93/strided_slice/stack:output:0.forward_lstm_93/strided_slice/stack_1:output:0.forward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros/packedPack&forward_lstm_93/strided_slice:output:0'forward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zerosFill%forward_lstm_93/zeros/packed:output:0$forward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:b
 forward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros_1/packedPack&forward_lstm_93/strided_slice:output:0)forward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zeros_1Fill'forward_lstm_93/zeros_1/packed:output:0&forward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:s
forward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose	Transposeinputs'forward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:l
forward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         o
%forward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_1StridedSlice forward_lstm_93/Shape_1:output:0.forward_lstm_93/strided_slice_1/stack:output:00forward_lstm_93/strided_slice_1/stack_1:output:00forward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
forward_lstm_93/TensorArrayV2TensorListReserve4forward_lstm_93/TensorArrayV2/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Eforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
7forward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_93/transpose:y:0Nforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���o
%forward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_2StridedSliceforward_lstm_93/transpose:y:0.forward_lstm_93/strided_slice_2/stack:output:00forward_lstm_93/strided_slice_2/stack_1:output:00forward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 forward_lstm_93/lstm_cell/MatMulMatMul(forward_lstm_93/strided_slice_2:output:07forward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
"forward_lstm_93/lstm_cell/MatMul_1MatMulforward_lstm_93/zeros:output:09forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
forward_lstm_93/lstm_cell/addAddV2*forward_lstm_93/lstm_cell/MatMul:product:0,forward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!forward_lstm_93/lstm_cell/BiasAddBiasAdd!forward_lstm_93/lstm_cell/add:z:08forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: k
)forward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/lstm_cell/splitSplit2forward_lstm_93/lstm_cell/split/split_dim:output:0*forward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split
!forward_lstm_93/lstm_cell/SigmoidSigmoid(forward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
#forward_lstm_93/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mulMul'forward_lstm_93/lstm_cell/Sigmoid_1:y:0 forward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:y
forward_lstm_93/lstm_cell/ReluRelu(forward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mul_1Mul%forward_lstm_93/lstm_cell/Sigmoid:y:0,forward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/add_1AddV2!forward_lstm_93/lstm_cell/mul:z:0#forward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
#forward_lstm_93/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:v
 forward_lstm_93/lstm_cell/Relu_1Relu#forward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mul_2Mul'forward_lstm_93/lstm_cell/Sigmoid_2:y:0.forward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:~
-forward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
forward_lstm_93/TensorArrayV2_1TensorListReserve6forward_lstm_93/TensorArrayV2_1/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���V
forward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������d
"forward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
forward_lstm_93/whileWhile+forward_lstm_93/while/loop_counter:output:01forward_lstm_93/while/maximum_iterations:output:0forward_lstm_93/time:output:0(forward_lstm_93/TensorArrayV2_1:handle:0forward_lstm_93/zeros:output:0 forward_lstm_93/zeros_1:output:0(forward_lstm_93/strided_slice_1:output:0Gforward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_93_lstm_cell_matmul_readvariableop_resource:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#forward_lstm_93_while_body_55913183*/
cond'R%
#forward_lstm_93_while_cond_55913182*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
@forward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
2forward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_93/while:output:3Iforward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0x
%forward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������q
'forward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_3StridedSlice;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_93/strided_slice_3/stack:output:00forward_lstm_93/strided_slice_3/stack_1:output:00forward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masku
 forward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose_1	Transpose;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:k
forward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
backward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         n
$backward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
backward_lstm_93/strided_sliceStridedSlicebackward_lstm_93/Shape:output:0-backward_lstm_93/strided_slice/stack:output:0/backward_lstm_93/strided_slice/stack_1:output:0/backward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros/packedPack'backward_lstm_93/strided_slice:output:0(backward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zerosFill&backward_lstm_93/zeros/packed:output:0%backward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:c
!backward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros_1/packedPack'backward_lstm_93/strided_slice:output:0*backward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zeros_1Fill(backward_lstm_93/zeros_1/packed:output:0'backward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:t
backward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose	Transposeinputs(backward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:m
backward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
&backward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_1StridedSlice!backward_lstm_93/Shape_1:output:0/backward_lstm_93/strided_slice_1/stack:output:01backward_lstm_93/strided_slice_1/stack_1:output:01backward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
backward_lstm_93/TensorArrayV2TensorListReserve5backward_lstm_93/TensorArrayV2/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���i
backward_lstm_93/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
backward_lstm_93/ReverseV2	ReverseV2backward_lstm_93/transpose:y:0(backward_lstm_93/ReverseV2/axis:output:0*
T0*"
_output_shapes
:�
Fbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
8backward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_93/ReverseV2:output:0Obackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���p
&backward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_2StridedSlicebackward_lstm_93/transpose:y:0/backward_lstm_93/strided_slice_2/stack:output:01backward_lstm_93/strided_slice_2/stack_1:output:01backward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
!backward_lstm_93/lstm_cell/MatMulMatMul)backward_lstm_93/strided_slice_2:output:08backward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
#backward_lstm_93/lstm_cell/MatMul_1MatMulbackward_lstm_93/zeros:output:0:backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
backward_lstm_93/lstm_cell/addAddV2+backward_lstm_93/lstm_cell/MatMul:product:0-backward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"backward_lstm_93/lstm_cell/BiasAddBiasAdd"backward_lstm_93/lstm_cell/add:z:09backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: l
*backward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 backward_lstm_93/lstm_cell/splitSplit3backward_lstm_93/lstm_cell/split/split_dim:output:0+backward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
"backward_lstm_93/lstm_cell/SigmoidSigmoid)backward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
$backward_lstm_93/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
backward_lstm_93/lstm_cell/mulMul(backward_lstm_93/lstm_cell/Sigmoid_1:y:0!backward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:{
backward_lstm_93/lstm_cell/ReluRelu)backward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/mul_1Mul&backward_lstm_93/lstm_cell/Sigmoid:y:0-backward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/add_1AddV2"backward_lstm_93/lstm_cell/mul:z:0$backward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
$backward_lstm_93/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:x
!backward_lstm_93/lstm_cell/Relu_1Relu$backward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/mul_2Mul(backward_lstm_93/lstm_cell/Sigmoid_2:y:0/backward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:
.backward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 backward_lstm_93/TensorArrayV2_1TensorListReserve7backward_lstm_93/TensorArrayV2_1/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���W
backward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������e
#backward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
backward_lstm_93/whileWhile,backward_lstm_93/while/loop_counter:output:02backward_lstm_93/while/maximum_iterations:output:0backward_lstm_93/time:output:0)backward_lstm_93/TensorArrayV2_1:handle:0backward_lstm_93/zeros:output:0!backward_lstm_93/zeros_1:output:0)backward_lstm_93/strided_slice_1:output:0Hbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_93_lstm_cell_matmul_readvariableop_resource;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$backward_lstm_93_while_body_55913324*0
cond(R&
$backward_lstm_93_while_cond_55913323*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
Abackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
3backward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_93/while:output:3Jbackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0y
&backward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������r
(backward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_3StridedSlice<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_93/strided_slice_3/stack:output:01backward_lstm_93/strided_slice_3/stack_1:output:01backward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskv
!backward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose_1	Transpose<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:l
backward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:~
	ReverseV2	ReverseV2 backward_lstm_93/transpose_1:y:0ReverseV2/axis:output:0*
T0*"
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2forward_lstm_93/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*"
_output_shapes
:Y
IdentityIdentityconcat:output:0^NoOp*
T0*"
_output_shapes
:�
NoOpNoOp2^backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_93/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_93/while1^forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_93/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2f
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_93/whilebackward_lstm_93/while2d
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_93/whileforward_lstm_93/while:J F
"
_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�J
�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914390

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55914306*
condR
while_cond_55914305*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�J
�
#forward_lstm_93_while_body_55913183<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3;
7forward_lstm_93_while_forward_lstm_93_strided_slice_1_0w
sforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_93_while_identity$
 forward_lstm_93_while_identity_1$
 forward_lstm_93_while_identity_2$
 forward_lstm_93_while_identity_3$
 forward_lstm_93_while_identity_4$
 forward_lstm_93_while_identity_59
5forward_lstm_93_while_forward_lstm_93_strided_slice_1u
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Gforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
9forward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_93_while_placeholderPforward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
&forward_lstm_93/while/lstm_cell/MatMulMatMul@forward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
(forward_lstm_93/while/lstm_cell/MatMul_1MatMul#forward_lstm_93_while_placeholder_2?forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
#forward_lstm_93/while/lstm_cell/addAddV20forward_lstm_93/while/lstm_cell/MatMul:product:02forward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
'forward_lstm_93/while/lstm_cell/BiasAddBiasAdd'forward_lstm_93/while/lstm_cell/add:z:0>forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
/forward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
%forward_lstm_93/while/lstm_cell/splitSplit8forward_lstm_93/while/lstm_cell/split/split_dim:output:00forward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
'forward_lstm_93/while/lstm_cell/SigmoidSigmoid.forward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
)forward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
#forward_lstm_93/while/lstm_cell/mulMul-forward_lstm_93/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
$forward_lstm_93/while/lstm_cell/ReluRelu.forward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/mul_1Mul+forward_lstm_93/while/lstm_cell/Sigmoid:y:02forward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/add_1AddV2'forward_lstm_93/while/lstm_cell/mul:z:0)forward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
)forward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
&forward_lstm_93/while/lstm_cell/Relu_1Relu)forward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
%forward_lstm_93/while/lstm_cell/mul_2Mul-forward_lstm_93/while/lstm_cell/Sigmoid_2:y:04forward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
:forward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_93_while_placeholder_1!forward_lstm_93_while_placeholder)forward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���]
forward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/addAddV2!forward_lstm_93_while_placeholder$forward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/while/add_1AddV28forward_lstm_93_while_forward_lstm_93_while_loop_counter&forward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_1Identity>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_2Identityforward_lstm_93/while/add:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_3IdentityJforward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
 forward_lstm_93/while/Identity_4Identity)forward_lstm_93/while/lstm_cell/mul_2:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
 forward_lstm_93/while/Identity_5Identity)forward_lstm_93/while/lstm_cell/add_1:z:0^forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
forward_lstm_93/while/NoOpNoOp7^forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_93_while_forward_lstm_93_strided_slice_17forward_lstm_93_while_forward_lstm_93_strided_slice_1_0"I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0"M
 forward_lstm_93_while_identity_1)forward_lstm_93/while/Identity_1:output:0"M
 forward_lstm_93_while_identity_2)forward_lstm_93/while/Identity_2:output:0"M
 forward_lstm_93_while_identity_3)forward_lstm_93/while/Identity_3:output:0"M
 forward_lstm_93_while_identity_4)forward_lstm_93/while/Identity_4:output:0"M
 forward_lstm_93_while_identity_5)forward_lstm_93/while/Identity_5:output:0"�
?forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
@forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
>forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
qforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorsforward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2p
6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
1__inference_sequential_408_layer_call_fn_55909738
dense_595_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_595_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909720s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_595_input:($
"
_user_specified_name
55909732:($
"
_user_specified_name
55909734
�
�
$backward_lstm_93_while_cond_55912041>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3@
<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55912041___redundant_placeholder0X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55912041___redundant_placeholder1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55912041___redundant_placeholder2X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55912041___redundant_placeholder3#
backward_lstm_93_while_identity
�
backward_lstm_93/while/LessLess"backward_lstm_93_while_placeholder<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_93/while/IdentityIdentitybackward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:

_output_shapes
:
�8
�
while_body_55914638
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915223

inputs
states_0
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910037

inputs

states
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910385

inputs

states
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
G__inference_dense_594_layer_call_and_return_conditional_losses_55911386

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     p
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Q
SoftmaxSoftmaxBiasAdd:output:0*
T0*"
_output_shapes
:[
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*"
_output_shapes
:V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�J
�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55910816

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55910732*
condR
while_cond_55910731*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�M
�
$backward_lstm_93_while_body_55913036>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3=
9backward_lstm_93_while_backward_lstm_93_strided_slice_1_0y
ubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_93_while_identity%
!backward_lstm_93_while_identity_1%
!backward_lstm_93_while_identity_2%
!backward_lstm_93_while_identity_3%
!backward_lstm_93_while_identity_4%
!backward_lstm_93_while_identity_5;
7backward_lstm_93_while_backward_lstm_93_strided_slice_1w
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
Hbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
:backward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_93_while_placeholderQbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
'backward_lstm_93/while/lstm_cell/MatMulMatMulAbackward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
)backward_lstm_93/while/lstm_cell/MatMul_1MatMul$backward_lstm_93_while_placeholder_2@backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
$backward_lstm_93/while/lstm_cell/addAddV21backward_lstm_93/while/lstm_cell/MatMul:product:03backward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
(backward_lstm_93/while/lstm_cell/BiasAddBiasAdd(backward_lstm_93/while/lstm_cell/add:z:0?backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� r
0backward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
&backward_lstm_93/while/lstm_cell/splitSplit9backward_lstm_93/while/lstm_cell/split/split_dim:output:01backward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
(backward_lstm_93/while/lstm_cell/SigmoidSigmoid/backward_lstm_93/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:����������
*backward_lstm_93/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_93/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
$backward_lstm_93/while/lstm_cell/mulMul.backward_lstm_93/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_93_while_placeholder_3*
T0*'
_output_shapes
:����������
%backward_lstm_93/while/lstm_cell/ReluRelu/backward_lstm_93/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
&backward_lstm_93/while/lstm_cell/mul_1Mul,backward_lstm_93/while/lstm_cell/Sigmoid:y:03backward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
&backward_lstm_93/while/lstm_cell/add_1AddV2(backward_lstm_93/while/lstm_cell/mul:z:0*backward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:����������
*backward_lstm_93/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_93/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:����������
'backward_lstm_93/while/lstm_cell/Relu_1Relu*backward_lstm_93/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
&backward_lstm_93/while/lstm_cell/mul_2Mul.backward_lstm_93/while/lstm_cell/Sigmoid_2:y:05backward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
;backward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_93_while_placeholder_1"backward_lstm_93_while_placeholder*backward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���^
backward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/addAddV2"backward_lstm_93_while_placeholder%backward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/while/add_1AddV2:backward_lstm_93_while_backward_lstm_93_while_loop_counter'backward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
backward_lstm_93/while/IdentityIdentity backward_lstm_93/while/add_1:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_1Identity@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_2Identitybackward_lstm_93/while/add:z:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_3IdentityKbackward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
!backward_lstm_93/while/Identity_4Identity*backward_lstm_93/while/lstm_cell/mul_2:z:0^backward_lstm_93/while/NoOp*
T0*'
_output_shapes
:����������
!backward_lstm_93/while/Identity_5Identity*backward_lstm_93/while/lstm_cell/add_1:z:0^backward_lstm_93/while/NoOp*
T0*'
_output_shapes
:����������
backward_lstm_93/while/NoOpNoOp8^backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_93_while_backward_lstm_93_strided_slice_19backward_lstm_93_while_backward_lstm_93_strided_slice_1_0"K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0"O
!backward_lstm_93_while_identity_1*backward_lstm_93/while/Identity_1:output:0"O
!backward_lstm_93_while_identity_2*backward_lstm_93/while/Identity_2:output:0"O
!backward_lstm_93_while_identity_3*backward_lstm_93/while/Identity_3:output:0"O
!backward_lstm_93_while_identity_4*backward_lstm_93/while/Identity_4:output:0"O
!backward_lstm_93_while_identity_5*backward_lstm_93/while/Identity_5:output:0"�
@backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
Abackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
?backward_lstm_93_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
sbackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorubackward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2r
7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
3__inference_bidirectional_93_layer_call_fn_55912514
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55911291|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55912500:($
"
_user_specified_name
55912502:($
"
_user_specified_name
55912504:($
"
_user_specified_name
55912506:($
"
_user_specified_name
55912508:($
"
_user_specified_name
55912510
�
�
,__inference_lstm_cell_layer_call_fn_55915174

inputs
states_0
states_1
unknown: 
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910037o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1:($
"
_user_specified_name
55915162:($
"
_user_specified_name
55915164:($
"
_user_specified_name
55915166
�8
�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55910120

inputs$
lstm_cell_55910038: $
lstm_cell_55910040:  
lstm_cell_55910042: 
identity��!lstm_cell/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55910038lstm_cell_55910040lstm_cell_55910042*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910037n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55910038lstm_cell_55910040lstm_cell_55910042*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55910051*
condR
while_cond_55910050*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55910038:($
"
_user_specified_name
55910040:($
"
_user_specified_name
55910042
ٖ
�
!__inference__traced_save_55915483
file_prefix9
'read_disablecopyonread_dense_594_kernel:5
'read_1_disablecopyonread_dense_594_bias:<
)read_2_disablecopyonread_dense_611_kernel:	�5
'read_3_disablecopyonread_dense_611_bias:;
)read_4_disablecopyonread_dense_595_kernel:5
'read_5_disablecopyonread_dense_595_bias:;
)read_6_disablecopyonread_dense_600_kernel:5
'read_7_disablecopyonread_dense_600_bias:;
)read_8_disablecopyonread_dense_605_kernel:5
'read_9_disablecopyonread_dense_605_bias:<
*read_10_disablecopyonread_dense_610_kernel:6
(read_11_disablecopyonread_dense_610_bias:]
Kread_12_disablecopyonread_bidirectional_93_forward_lstm_93_lstm_cell_kernel: g
Uread_13_disablecopyonread_bidirectional_93_forward_lstm_93_lstm_cell_recurrent_kernel: W
Iread_14_disablecopyonread_bidirectional_93_forward_lstm_93_lstm_cell_bias: ^
Lread_15_disablecopyonread_bidirectional_93_backward_lstm_93_lstm_cell_kernel: h
Vread_16_disablecopyonread_bidirectional_93_backward_lstm_93_lstm_cell_recurrent_kernel: X
Jread_17_disablecopyonread_bidirectional_93_backward_lstm_93_lstm_cell_bias: 
savev2_const
identity_37��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_dense_594_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_dense_594_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_dense_594_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_dense_594_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_dense_611_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_dense_611_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_dense_611_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_dense_611_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_4/DisableCopyOnReadDisableCopyOnRead)read_4_disablecopyonread_dense_595_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp)read_4_disablecopyonread_dense_595_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:{
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_dense_595_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_dense_595_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_dense_600_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_dense_600_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:{
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_dense_600_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_dense_600_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:}
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_dense_605_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_dense_605_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:{
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_dense_605_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_dense_605_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_10/DisableCopyOnReadDisableCopyOnRead*read_10_disablecopyonread_dense_610_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_dense_610_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:}
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_dense_610_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_dense_610_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_12/DisableCopyOnReadDisableCopyOnReadKread_12_disablecopyonread_bidirectional_93_forward_lstm_93_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOpKread_12_disablecopyonread_bidirectional_93_forward_lstm_93_lstm_cell_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_13/DisableCopyOnReadDisableCopyOnReadUread_13_disablecopyonread_bidirectional_93_forward_lstm_93_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOpUread_13_disablecopyonread_bidirectional_93_forward_lstm_93_lstm_cell_recurrent_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_14/DisableCopyOnReadDisableCopyOnReadIread_14_disablecopyonread_bidirectional_93_forward_lstm_93_lstm_cell_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpIread_14_disablecopyonread_bidirectional_93_forward_lstm_93_lstm_cell_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_15/DisableCopyOnReadDisableCopyOnReadLread_15_disablecopyonread_bidirectional_93_backward_lstm_93_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOpLread_15_disablecopyonread_bidirectional_93_backward_lstm_93_lstm_cell_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_16/DisableCopyOnReadDisableCopyOnReadVread_16_disablecopyonread_bidirectional_93_backward_lstm_93_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpVread_16_disablecopyonread_bidirectional_93_backward_lstm_93_lstm_cell_recurrent_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

: �
Read_17/DisableCopyOnReadDisableCopyOnReadJread_17_disablecopyonread_bidirectional_93_backward_lstm_93_lstm_cell_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpJread_17_disablecopyonread_bidirectional_93_backward_lstm_93_lstm_cell_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_36Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_37IdentityIdentity_36:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_37Identity_37:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:0,
*
_user_specified_namedense_594/kernel:.*
(
_user_specified_namedense_594/bias:0,
*
_user_specified_namedense_611/kernel:.*
(
_user_specified_namedense_611/bias:0,
*
_user_specified_namedense_595/kernel:.*
(
_user_specified_namedense_595/bias:0,
*
_user_specified_namedense_600/kernel:.*
(
_user_specified_namedense_600/bias:0	,
*
_user_specified_namedense_605/kernel:.
*
(
_user_specified_namedense_605/bias:0,
*
_user_specified_namedense_610/kernel:.*
(
_user_specified_namedense_610/bias:QM
K
_user_specified_name31bidirectional_93/forward_lstm_93/lstm_cell/kernel:[W
U
_user_specified_name=;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel:OK
I
_user_specified_name1/bidirectional_93/forward_lstm_93/lstm_cell/bias:RN
L
_user_specified_name42bidirectional_93/backward_lstm_93/lstm_cell/kernel:\X
V
_user_specified_name><bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel:PL
J
_user_specified_name20bidirectional_93/backward_lstm_93/lstm_cell/bias:=9

_output_shapes
: 

_user_specified_nameConst
�	
�
2__inference_forward_lstm_93_layer_call_fn_55913939
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55910265|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55913931:($
"
_user_specified_name
55913933:($
"
_user_specified_name
55913935
�
�
#forward_lstm_93_while_cond_55911502<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3>
:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55911502___redundant_placeholder0V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55911502___redundant_placeholder1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55911502___redundant_placeholder2V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55911502___redundant_placeholder3"
forward_lstm_93_while_identity
�
forward_lstm_93/while/LessLess!forward_lstm_93_while_placeholder:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:

_output_shapes
:
�:
�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55910468

inputs$
lstm_cell_55910386: $
lstm_cell_55910388:  
lstm_cell_55910390: 
identity��!lstm_cell/StatefulPartitionedCall�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :�������������������
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55910386lstm_cell_55910388lstm_cell_55910390*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910385n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55910386lstm_cell_55910388lstm_cell_55910390*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55910399*
condR
while_cond_55910398*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55910386:($
"
_user_specified_name
55910388:($
"
_user_specified_name
55910390
�L
�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55915157

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'����������������������������
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55915073*
condR
while_cond_55915072*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�8
�
while_body_55910884
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
G__inference_dense_600_layer_call_and_return_conditional_losses_55909780

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
3__inference_bidirectional_93_layer_call_fn_55912497
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55910981|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55912483:($
"
_user_specified_name
55912485:($
"
_user_specified_name
55912487:($
"
_user_specified_name
55912489:($
"
_user_specified_name
55912491:($
"
_user_specified_name
55912493
�
d
H__inference_flatten_93_layer_call_and_return_conditional_losses_55911764

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  T
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	�P
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�8
�
while_body_55914783
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
��
�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55911732

inputsJ
8forward_lstm_93_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_93_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource: 
identity��1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp�2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�backward_lstm_93/while�0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp�1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�forward_lstm_93/whilej
forward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         m
#forward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_sliceStridedSliceforward_lstm_93/Shape:output:0,forward_lstm_93/strided_slice/stack:output:0.forward_lstm_93/strided_slice/stack_1:output:0.forward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros/packedPack&forward_lstm_93/strided_slice:output:0'forward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zerosFill%forward_lstm_93/zeros/packed:output:0$forward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:b
 forward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros_1/packedPack&forward_lstm_93/strided_slice:output:0)forward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zeros_1Fill'forward_lstm_93/zeros_1/packed:output:0&forward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:s
forward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose	Transposeinputs'forward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:l
forward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         o
%forward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_1StridedSlice forward_lstm_93/Shape_1:output:0.forward_lstm_93/strided_slice_1/stack:output:00forward_lstm_93/strided_slice_1/stack_1:output:00forward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
forward_lstm_93/TensorArrayV2TensorListReserve4forward_lstm_93/TensorArrayV2/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Eforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
7forward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_93/transpose:y:0Nforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���o
%forward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_2StridedSliceforward_lstm_93/transpose:y:0.forward_lstm_93/strided_slice_2/stack:output:00forward_lstm_93/strided_slice_2/stack_1:output:00forward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 forward_lstm_93/lstm_cell/MatMulMatMul(forward_lstm_93/strided_slice_2:output:07forward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
"forward_lstm_93/lstm_cell/MatMul_1MatMulforward_lstm_93/zeros:output:09forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
forward_lstm_93/lstm_cell/addAddV2*forward_lstm_93/lstm_cell/MatMul:product:0,forward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!forward_lstm_93/lstm_cell/BiasAddBiasAdd!forward_lstm_93/lstm_cell/add:z:08forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: k
)forward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/lstm_cell/splitSplit2forward_lstm_93/lstm_cell/split/split_dim:output:0*forward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split
!forward_lstm_93/lstm_cell/SigmoidSigmoid(forward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
#forward_lstm_93/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mulMul'forward_lstm_93/lstm_cell/Sigmoid_1:y:0 forward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:y
forward_lstm_93/lstm_cell/ReluRelu(forward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mul_1Mul%forward_lstm_93/lstm_cell/Sigmoid:y:0,forward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/add_1AddV2!forward_lstm_93/lstm_cell/mul:z:0#forward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
#forward_lstm_93/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:v
 forward_lstm_93/lstm_cell/Relu_1Relu#forward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mul_2Mul'forward_lstm_93/lstm_cell/Sigmoid_2:y:0.forward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:~
-forward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
forward_lstm_93/TensorArrayV2_1TensorListReserve6forward_lstm_93/TensorArrayV2_1/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���V
forward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������d
"forward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
forward_lstm_93/whileWhile+forward_lstm_93/while/loop_counter:output:01forward_lstm_93/while/maximum_iterations:output:0forward_lstm_93/time:output:0(forward_lstm_93/TensorArrayV2_1:handle:0forward_lstm_93/zeros:output:0 forward_lstm_93/zeros_1:output:0(forward_lstm_93/strided_slice_1:output:0Gforward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_93_lstm_cell_matmul_readvariableop_resource:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#forward_lstm_93_while_body_55911503*/
cond'R%
#forward_lstm_93_while_cond_55911502*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
@forward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
2forward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_93/while:output:3Iforward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0x
%forward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������q
'forward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_3StridedSlice;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_93/strided_slice_3/stack:output:00forward_lstm_93/strided_slice_3/stack_1:output:00forward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masku
 forward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose_1	Transpose;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:k
forward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
backward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         n
$backward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
backward_lstm_93/strided_sliceStridedSlicebackward_lstm_93/Shape:output:0-backward_lstm_93/strided_slice/stack:output:0/backward_lstm_93/strided_slice/stack_1:output:0/backward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros/packedPack'backward_lstm_93/strided_slice:output:0(backward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zerosFill&backward_lstm_93/zeros/packed:output:0%backward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:c
!backward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros_1/packedPack'backward_lstm_93/strided_slice:output:0*backward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zeros_1Fill(backward_lstm_93/zeros_1/packed:output:0'backward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:t
backward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose	Transposeinputs(backward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:m
backward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
&backward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_1StridedSlice!backward_lstm_93/Shape_1:output:0/backward_lstm_93/strided_slice_1/stack:output:01backward_lstm_93/strided_slice_1/stack_1:output:01backward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
backward_lstm_93/TensorArrayV2TensorListReserve5backward_lstm_93/TensorArrayV2/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���i
backward_lstm_93/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
backward_lstm_93/ReverseV2	ReverseV2backward_lstm_93/transpose:y:0(backward_lstm_93/ReverseV2/axis:output:0*
T0*"
_output_shapes
:�
Fbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
8backward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_93/ReverseV2:output:0Obackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���p
&backward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_2StridedSlicebackward_lstm_93/transpose:y:0/backward_lstm_93/strided_slice_2/stack:output:01backward_lstm_93/strided_slice_2/stack_1:output:01backward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
!backward_lstm_93/lstm_cell/MatMulMatMul)backward_lstm_93/strided_slice_2:output:08backward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
#backward_lstm_93/lstm_cell/MatMul_1MatMulbackward_lstm_93/zeros:output:0:backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
backward_lstm_93/lstm_cell/addAddV2+backward_lstm_93/lstm_cell/MatMul:product:0-backward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"backward_lstm_93/lstm_cell/BiasAddBiasAdd"backward_lstm_93/lstm_cell/add:z:09backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: l
*backward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 backward_lstm_93/lstm_cell/splitSplit3backward_lstm_93/lstm_cell/split/split_dim:output:0+backward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
"backward_lstm_93/lstm_cell/SigmoidSigmoid)backward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
$backward_lstm_93/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
backward_lstm_93/lstm_cell/mulMul(backward_lstm_93/lstm_cell/Sigmoid_1:y:0!backward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:{
backward_lstm_93/lstm_cell/ReluRelu)backward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/mul_1Mul&backward_lstm_93/lstm_cell/Sigmoid:y:0-backward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/add_1AddV2"backward_lstm_93/lstm_cell/mul:z:0$backward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
$backward_lstm_93/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:x
!backward_lstm_93/lstm_cell/Relu_1Relu$backward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/mul_2Mul(backward_lstm_93/lstm_cell/Sigmoid_2:y:0/backward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:
.backward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 backward_lstm_93/TensorArrayV2_1TensorListReserve7backward_lstm_93/TensorArrayV2_1/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���W
backward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������e
#backward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
backward_lstm_93/whileWhile,backward_lstm_93/while/loop_counter:output:02backward_lstm_93/while/maximum_iterations:output:0backward_lstm_93/time:output:0)backward_lstm_93/TensorArrayV2_1:handle:0backward_lstm_93/zeros:output:0!backward_lstm_93/zeros_1:output:0)backward_lstm_93/strided_slice_1:output:0Hbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_93_lstm_cell_matmul_readvariableop_resource;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$backward_lstm_93_while_body_55911644*0
cond(R&
$backward_lstm_93_while_cond_55911643*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
Abackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
3backward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_93/while:output:3Jbackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0y
&backward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������r
(backward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_3StridedSlice<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_93/strided_slice_3/stack:output:01backward_lstm_93/strided_slice_3/stack_1:output:01backward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskv
!backward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose_1	Transpose<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:l
backward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:~
	ReverseV2	ReverseV2 backward_lstm_93/transpose_1:y:0ReverseV2/axis:output:0*
T0*"
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2forward_lstm_93/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*"
_output_shapes
:Y
IdentityIdentityconcat:output:0^NoOp*
T0*"
_output_shapes
:�
NoOpNoOp2^backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_93/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_93/while1^forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_93/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2f
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_93/whilebackward_lstm_93/while2d
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_93/whileforward_lstm_93/while:J F
"
_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
while_cond_55915072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55915072___redundant_placeholder06
2while_while_cond_55915072___redundant_placeholder16
2while_while_cond_55915072___redundant_placeholder26
2while_while_cond_55915072___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
�
G__inference_dense_595_layer_call_and_return_conditional_losses_55913797

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
r
H__inference_lambda_101_layer_call_and_return_conditional_losses_55911841

inputs
inputs_1
identity|
einsum/EinsumEinsuminputsinputs_1*
N*
T0*"
_output_shapes
:*
equationabcd,ace->acdY
IdentityIdentityeinsum/Einsum:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::N J
&
_output_shapes
:
 
_user_specified_nameinputs:JF
"
_output_shapes
:
 
_user_specified_nameinputs
�	
�
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909796
dense_600_input$
dense_600_55909790: 
dense_600_55909792:
identity��!dense_600/StatefulPartitionedCall�
!dense_600/StatefulPartitionedCallStatefulPartitionedCalldense_600_inputdense_600_55909790dense_600_55909792*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_600_layer_call_and_return_conditional_losses_55909780}
IdentityIdentity*dense_600/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������F
NoOpNoOp"^dense_600/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2F
!dense_600/StatefulPartitionedCall!dense_600/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_600_input:($
"
_user_specified_name
55909790:($
"
_user_specified_name
55909792
�
�
G__inference_dense_610_layer_call_and_return_conditional_losses_55909932

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�L
�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55910968

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'����������������������������
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55910884*
condR
while_cond_55910883*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�$
�
while_body_55910546
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_55910570_0: ,
while_lstm_cell_55910572_0: (
while_lstm_cell_55910574_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_55910570: *
while_lstm_cell_55910572: &
while_lstm_cell_55910574: ��'while/lstm_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55910570_0while_lstm_cell_55910572_0while_lstm_cell_55910574_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910532�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_55910570while_lstm_cell_55910570_0"6
while_lstm_cell_55910572while_lstm_cell_55910572_0"6
while_lstm_cell_55910574while_lstm_cell_55910574_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
55910570:(	$
"
_user_specified_name
55910572:(
$
"
_user_specified_name
55910574
�
�
#forward_lstm_93_while_cond_55912894<
8forward_lstm_93_while_forward_lstm_93_while_loop_counterB
>forward_lstm_93_while_forward_lstm_93_while_maximum_iterations%
!forward_lstm_93_while_placeholder'
#forward_lstm_93_while_placeholder_1'
#forward_lstm_93_while_placeholder_2'
#forward_lstm_93_while_placeholder_3>
:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55912894___redundant_placeholder0V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55912894___redundant_placeholder1V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55912894___redundant_placeholder2V
Rforward_lstm_93_while_forward_lstm_93_while_cond_55912894___redundant_placeholder3"
forward_lstm_93_while_identity
�
forward_lstm_93/while/LessLess!forward_lstm_93_while_placeholder:forward_lstm_93_while_less_forward_lstm_93_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_93/while/IdentityIdentityforward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_93_while_identity'forward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_93/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_93/strided_slice_1:

_output_shapes
:
�8
�
while_body_55914928
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�K
�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55914867
inputs_0:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :�������������������
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55914783*
condR
while_cond_55914782*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
1__inference_sequential_423_layer_call_fn_55909966
dense_610_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_610_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909948s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_610_input:($
"
_user_specified_name
55909960:($
"
_user_specified_name
55909962
�$
�
while_body_55910399
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_55910423_0: ,
while_lstm_cell_55910425_0: (
while_lstm_cell_55910427_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_55910423: *
while_lstm_cell_55910425: &
while_lstm_cell_55910427: ��'while/lstm_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55910423_0while_lstm_cell_55910425_0while_lstm_cell_55910427_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910385�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_55910423while_lstm_cell_55910423_0"6
while_lstm_cell_55910425while_lstm_cell_55910425_0"6
while_lstm_cell_55910427while_lstm_cell_55910427_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
55910423:(	$
"
_user_specified_name
55910425:(
$
"
_user_specified_name
55910427
�$
�
while_body_55910196
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_55910220_0: ,
while_lstm_cell_55910222_0: (
while_lstm_cell_55910224_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_55910220: *
while_lstm_cell_55910222: &
while_lstm_cell_55910224: ��'while/lstm_cell/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0�
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55910220_0while_lstm_cell_55910222_0while_lstm_cell_55910224_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910182�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:����������
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:���������R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_55910220while_lstm_cell_55910220_0"6
while_lstm_cell_55910222while_lstm_cell_55910222_0"6
while_lstm_cell_55910224while_lstm_cell_55910224_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
55910220:(	$
"
_user_specified_name
55910222:(
$
"
_user_specified_name
55910224
�K
�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55914722
inputs_0:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :�������������������
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55914638*
condR
while_cond_55914637*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
G__inference_dense_611_layer_call_and_return_conditional_losses_55913757

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	�
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
while_cond_55910731
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55910731___redundant_placeholder06
2while_while_cond_55910731___redundant_placeholder16
2while_while_cond_55910731___redundant_placeholder26
2while_while_cond_55910731___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�	
�
2__inference_forward_lstm_93_layer_call_fn_55913950

inputs
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55910816|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55913942:($
"
_user_specified_name
55913944:($
"
_user_specified_name
55913946
�
X
,__inference_lambda_99_layer_call_fn_55912420
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lambda_99_layer_call_and_return_conditional_losses_55911802[
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::L H
"
_output_shapes
:
"
_user_specified_name
inputs_0:PL
&
_output_shapes
:
"
_user_specified_name
inputs_1
�

g
H__inference_dropout_93_layer_call_and_return_conditional_losses_55911757

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
:b
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
:*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*"
_output_shapes
:\
IdentityIdentitydropout/SelectV2:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
��
�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913700

inputsJ
8forward_lstm_93_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_93_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource: 
identity��1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp�2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�backward_lstm_93/while�0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp�1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�forward_lstm_93/whilej
forward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         m
#forward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_sliceStridedSliceforward_lstm_93/Shape:output:0,forward_lstm_93/strided_slice/stack:output:0.forward_lstm_93/strided_slice/stack_1:output:0.forward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros/packedPack&forward_lstm_93/strided_slice:output:0'forward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zerosFill%forward_lstm_93/zeros/packed:output:0$forward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:b
 forward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros_1/packedPack&forward_lstm_93/strided_slice:output:0)forward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zeros_1Fill'forward_lstm_93/zeros_1/packed:output:0&forward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:s
forward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose	Transposeinputs'forward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:l
forward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         o
%forward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_1StridedSlice forward_lstm_93/Shape_1:output:0.forward_lstm_93/strided_slice_1/stack:output:00forward_lstm_93/strided_slice_1/stack_1:output:00forward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
forward_lstm_93/TensorArrayV2TensorListReserve4forward_lstm_93/TensorArrayV2/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Eforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
7forward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_93/transpose:y:0Nforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���o
%forward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_2StridedSliceforward_lstm_93/transpose:y:0.forward_lstm_93/strided_slice_2/stack:output:00forward_lstm_93/strided_slice_2/stack_1:output:00forward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 forward_lstm_93/lstm_cell/MatMulMatMul(forward_lstm_93/strided_slice_2:output:07forward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
"forward_lstm_93/lstm_cell/MatMul_1MatMulforward_lstm_93/zeros:output:09forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
forward_lstm_93/lstm_cell/addAddV2*forward_lstm_93/lstm_cell/MatMul:product:0,forward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!forward_lstm_93/lstm_cell/BiasAddBiasAdd!forward_lstm_93/lstm_cell/add:z:08forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: k
)forward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/lstm_cell/splitSplit2forward_lstm_93/lstm_cell/split/split_dim:output:0*forward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split
!forward_lstm_93/lstm_cell/SigmoidSigmoid(forward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
#forward_lstm_93/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mulMul'forward_lstm_93/lstm_cell/Sigmoid_1:y:0 forward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:y
forward_lstm_93/lstm_cell/ReluRelu(forward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mul_1Mul%forward_lstm_93/lstm_cell/Sigmoid:y:0,forward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/add_1AddV2!forward_lstm_93/lstm_cell/mul:z:0#forward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
#forward_lstm_93/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:v
 forward_lstm_93/lstm_cell/Relu_1Relu#forward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
forward_lstm_93/lstm_cell/mul_2Mul'forward_lstm_93/lstm_cell/Sigmoid_2:y:0.forward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:~
-forward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
forward_lstm_93/TensorArrayV2_1TensorListReserve6forward_lstm_93/TensorArrayV2_1/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���V
forward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������d
"forward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
forward_lstm_93/whileWhile+forward_lstm_93/while/loop_counter:output:01forward_lstm_93/while/maximum_iterations:output:0forward_lstm_93/time:output:0(forward_lstm_93/TensorArrayV2_1:handle:0forward_lstm_93/zeros:output:0 forward_lstm_93/zeros_1:output:0(forward_lstm_93/strided_slice_1:output:0Gforward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_93_lstm_cell_matmul_readvariableop_resource:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#forward_lstm_93_while_body_55913471*/
cond'R%
#forward_lstm_93_while_cond_55913470*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
@forward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
2forward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_93/while:output:3Iforward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0x
%forward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������q
'forward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_3StridedSlice;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_93/strided_slice_3/stack:output:00forward_lstm_93/strided_slice_3/stack_1:output:00forward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masku
 forward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose_1	Transpose;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:k
forward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
backward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         n
$backward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
backward_lstm_93/strided_sliceStridedSlicebackward_lstm_93/Shape:output:0-backward_lstm_93/strided_slice/stack:output:0/backward_lstm_93/strided_slice/stack_1:output:0/backward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros/packedPack'backward_lstm_93/strided_slice:output:0(backward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zerosFill&backward_lstm_93/zeros/packed:output:0%backward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:c
!backward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros_1/packedPack'backward_lstm_93/strided_slice:output:0*backward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zeros_1Fill(backward_lstm_93/zeros_1/packed:output:0'backward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:t
backward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose	Transposeinputs(backward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:m
backward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
&backward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_1StridedSlice!backward_lstm_93/Shape_1:output:0/backward_lstm_93/strided_slice_1/stack:output:01backward_lstm_93/strided_slice_1/stack_1:output:01backward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
backward_lstm_93/TensorArrayV2TensorListReserve5backward_lstm_93/TensorArrayV2/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���i
backward_lstm_93/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
backward_lstm_93/ReverseV2	ReverseV2backward_lstm_93/transpose:y:0(backward_lstm_93/ReverseV2/axis:output:0*
T0*"
_output_shapes
:�
Fbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
8backward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_93/ReverseV2:output:0Obackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���p
&backward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_2StridedSlicebackward_lstm_93/transpose:y:0/backward_lstm_93/strided_slice_2/stack:output:01backward_lstm_93/strided_slice_2/stack_1:output:01backward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
!backward_lstm_93/lstm_cell/MatMulMatMul)backward_lstm_93/strided_slice_2:output:08backward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
#backward_lstm_93/lstm_cell/MatMul_1MatMulbackward_lstm_93/zeros:output:0:backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
backward_lstm_93/lstm_cell/addAddV2+backward_lstm_93/lstm_cell/MatMul:product:0-backward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"backward_lstm_93/lstm_cell/BiasAddBiasAdd"backward_lstm_93/lstm_cell/add:z:09backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: l
*backward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 backward_lstm_93/lstm_cell/splitSplit3backward_lstm_93/lstm_cell/split/split_dim:output:0+backward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
"backward_lstm_93/lstm_cell/SigmoidSigmoid)backward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
$backward_lstm_93/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
backward_lstm_93/lstm_cell/mulMul(backward_lstm_93/lstm_cell/Sigmoid_1:y:0!backward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:{
backward_lstm_93/lstm_cell/ReluRelu)backward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/mul_1Mul&backward_lstm_93/lstm_cell/Sigmoid:y:0-backward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/add_1AddV2"backward_lstm_93/lstm_cell/mul:z:0$backward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
$backward_lstm_93/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:x
!backward_lstm_93/lstm_cell/Relu_1Relu$backward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
 backward_lstm_93/lstm_cell/mul_2Mul(backward_lstm_93/lstm_cell/Sigmoid_2:y:0/backward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:
.backward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
 backward_lstm_93/TensorArrayV2_1TensorListReserve7backward_lstm_93/TensorArrayV2_1/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���W
backward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������e
#backward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
backward_lstm_93/whileWhile,backward_lstm_93/while/loop_counter:output:02backward_lstm_93/while/maximum_iterations:output:0backward_lstm_93/time:output:0)backward_lstm_93/TensorArrayV2_1:handle:0backward_lstm_93/zeros:output:0!backward_lstm_93/zeros_1:output:0)backward_lstm_93/strided_slice_1:output:0Hbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_93_lstm_cell_matmul_readvariableop_resource;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$backward_lstm_93_while_body_55913612*0
cond(R&
$backward_lstm_93_while_cond_55913611*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
Abackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
3backward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_93/while:output:3Jbackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0y
&backward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������r
(backward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_3StridedSlice<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_93/strided_slice_3/stack:output:01backward_lstm_93/strided_slice_3/stack_1:output:01backward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskv
!backward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose_1	Transpose<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:l
backward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:~
	ReverseV2	ReverseV2 backward_lstm_93/transpose_1:y:0ReverseV2/axis:output:0*
T0*"
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2forward_lstm_93/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*"
_output_shapes
:Y
IdentityIdentityconcat:output:0^NoOp*
T0*"
_output_shapes
:�
NoOpNoOp2^backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_93/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_93/while1^forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_93/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2f
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_93/whilebackward_lstm_93/while2d
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_93/whileforward_lstm_93/while:J F
"
_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909787
dense_600_input$
dense_600_55909781: 
dense_600_55909783:
identity��!dense_600/StatefulPartitionedCall�
!dense_600/StatefulPartitionedCallStatefulPartitionedCalldense_600_inputdense_600_55909781dense_600_55909783*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_600_layer_call_and_return_conditional_losses_55909780}
IdentityIdentity*dense_600/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������F
NoOpNoOp"^dense_600/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2F
!dense_600/StatefulPartitionedCall!dense_600/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_600_input:($
"
_user_specified_name
55909781:($
"
_user_specified_name
55909783
�
�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915255

inputs
states_0
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909863
dense_605_input$
dense_605_55909857: 
dense_605_55909859:
identity��!dense_605/StatefulPartitionedCall�
!dense_605/StatefulPartitionedCallStatefulPartitionedCalldense_605_inputdense_605_55909857dense_605_55909859*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_605_layer_call_and_return_conditional_losses_55909856}
IdentityIdentity*dense_605/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������F
NoOpNoOp"^dense_605/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2F
!dense_605/StatefulPartitionedCall!dense_605/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_605_input:($
"
_user_specified_name
55909857:($
"
_user_specified_name
55909859
�
�
$backward_lstm_93_while_cond_55913035>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3@
<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913035___redundant_placeholder0X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913035___redundant_placeholder1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913035___redundant_placeholder2X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913035___redundant_placeholder3#
backward_lstm_93_while_identity
�
backward_lstm_93/while/LessLess"backward_lstm_93_while_placeholder<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_93/while/IdentityIdentitybackward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:

_output_shapes
:
�G
�
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55912157
input_4$
dense_594_55911785: 
dense_594_55911787:)
sequential_408_55911815:%
sequential_408_55911817:)
sequential_413_55911820:%
sequential_413_55911822:)
sequential_418_55911825:%
sequential_418_55911827:)
sequential_423_55911830:%
sequential_423_55911832:+
bidirectional_93_55912131: +
bidirectional_93_55912133: '
bidirectional_93_55912135: +
bidirectional_93_55912137: +
bidirectional_93_55912139: '
bidirectional_93_55912141: %
dense_611_55912151:	� 
dense_611_55912153:
identity��(bidirectional_93/StatefulPartitionedCall�!dense_594/StatefulPartitionedCall�!dense_611/StatefulPartitionedCall�&sequential_408/StatefulPartitionedCall�&sequential_413/StatefulPartitionedCall�&sequential_418/StatefulPartitionedCall�&sequential_423/StatefulPartitionedCall�
!dense_594/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_594_55911785dense_594_55911787*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_594_layer_call_and_return_conditional_losses_55911386Z
tf.math.top_k_3/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.top_k_3/TopKV2TopKV2*dense_594/StatefulPartitionedCall:output:0!tf.math.top_k_3/TopKV2/k:output:0*
T0*0
_output_shapes
::b
tf.one_hot_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?c
tf.one_hot_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    \
tf.one_hot_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
tf.one_hot_3/one_hotOneHot tf.math.top_k_3/TopKV2:indices:0#tf.one_hot_3/one_hot/depth:output:0&tf.one_hot_3/one_hot/on_value:output:0'tf.one_hot_3/one_hot/off_value:output:0*
T0*
TI0*&
_output_shapes
:�
lambda_99/PartitionedCallPartitionedCalltf.math.top_k_3/TopKV2:values:0tf.one_hot_3/one_hot:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lambda_99_layer_call_and_return_conditional_losses_55911802�
lambda_100/PartitionedCallPartitionedCallinput_4"lambda_99/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lambda_100_layer_call_and_return_conditional_losses_55911809�
tf.unstack_3/unstackUnpack#lambda_100/PartitionedCall:output:0*
T0*L
_output_shapes:
8::::*	
num�
&sequential_408/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_3/unstack:output:0sequential_408_55911815sequential_408_55911817*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909720�
&sequential_413/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_3/unstack:output:1sequential_413_55911820sequential_413_55911822*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909796�
&sequential_418/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_3/unstack:output:2sequential_418_55911825sequential_418_55911827*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909872�
&sequential_423/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_3/unstack:output:3sequential_423_55911830sequential_423_55911832*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909948�
tf.stack_94/stackPack/sequential_408/StatefulPartitionedCall:output:0/sequential_413/StatefulPartitionedCall:output:0/sequential_418/StatefulPartitionedCall:output:0/sequential_423/StatefulPartitionedCall:output:0*
N*
T0*&
_output_shapes
:*

axis�
lambda_101/PartitionedCallPartitionedCalltf.stack_94/stack:output:0"lambda_99/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lambda_101_layer_call_and_return_conditional_losses_55911841�
(bidirectional_93/StatefulPartitionedCallStatefulPartitionedCall#lambda_101/PartitionedCall:output:0bidirectional_93_55912131bidirectional_93_55912133bidirectional_93_55912135bidirectional_93_55912137bidirectional_93_55912139bidirectional_93_55912141*
Tin
	2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55912130�
dropout_93/PartitionedCallPartitionedCall1bidirectional_93/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_93_layer_call_and_return_conditional_losses_55912148�
flatten_93/PartitionedCallPartitionedCall#dropout_93/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_flatten_93_layer_call_and_return_conditional_losses_55911764�
!dense_611/StatefulPartitionedCallStatefulPartitionedCall#flatten_93/PartitionedCall:output:0dense_611_55912151dense_611_55912153*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_611_layer_call_and_return_conditional_losses_55911775p
IdentityIdentity*dense_611/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp)^bidirectional_93/StatefulPartitionedCall"^dense_594/StatefulPartitionedCall"^dense_611/StatefulPartitionedCall'^sequential_408/StatefulPartitionedCall'^sequential_413/StatefulPartitionedCall'^sequential_418/StatefulPartitionedCall'^sequential_423/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 2T
(bidirectional_93/StatefulPartitionedCall(bidirectional_93/StatefulPartitionedCall2F
!dense_594/StatefulPartitionedCall!dense_594/StatefulPartitionedCall2F
!dense_611/StatefulPartitionedCall!dense_611/StatefulPartitionedCall2P
&sequential_408/StatefulPartitionedCall&sequential_408/StatefulPartitionedCall2P
&sequential_413/StatefulPartitionedCall&sequential_413/StatefulPartitionedCall2P
&sequential_418/StatefulPartitionedCall&sequential_418/StatefulPartitionedCall2P
&sequential_423/StatefulPartitionedCall&sequential_423/StatefulPartitionedCall:K G
"
_output_shapes
:
!
_user_specified_name	input_4:($
"
_user_specified_name
55911785:($
"
_user_specified_name
55911787:($
"
_user_specified_name
55911815:($
"
_user_specified_name
55911817:($
"
_user_specified_name
55911820:($
"
_user_specified_name
55911822:($
"
_user_specified_name
55911825:($
"
_user_specified_name
55911827:(	$
"
_user_specified_name
55911830:(
$
"
_user_specified_name
55911832:($
"
_user_specified_name
55912131:($
"
_user_specified_name
55912133:($
"
_user_specified_name
55912135:($
"
_user_specified_name
55912137:($
"
_user_specified_name
55912139:($
"
_user_specified_name
55912141:($
"
_user_specified_name
55912151:($
"
_user_specified_name
55912153
�
s
G__inference_lambda_99_layer_call_and_return_conditional_losses_55912432
inputs_0
inputs_1
identity~
einsum/EinsumEinsuminputs_0inputs_1*
N*
T0*"
_output_shapes
:*
equationabc,abcd->abdY
IdentityIdentityeinsum/Einsum:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::L H
"
_output_shapes
:
"
_user_specified_name
inputs_0:PL
&
_output_shapes
:
"
_user_specified_name
inputs_1
�	
�
3__inference_backward_lstm_93_layer_call_fn_55914566

inputs
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55910968|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55914558:($
"
_user_specified_name
55914560:($
"
_user_specified_name
55914562
�
�
,__inference_lstm_cell_layer_call_fn_55915289

inputs
states_0
states_1
unknown: 
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910532o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1:($
"
_user_specified_name
55915277:($
"
_user_specified_name
55915279:($
"
_user_specified_name
55915281
�
�
,__inference_dense_600_layer_call_fn_55913806

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_600_layer_call_and_return_conditional_losses_55909780s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
55913800:($
"
_user_specified_name
55913802
�
�
$backward_lstm_93_while_cond_55913323>
:backward_lstm_93_while_backward_lstm_93_while_loop_counterD
@backward_lstm_93_while_backward_lstm_93_while_maximum_iterations&
"backward_lstm_93_while_placeholder(
$backward_lstm_93_while_placeholder_1(
$backward_lstm_93_while_placeholder_2(
$backward_lstm_93_while_placeholder_3@
<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913323___redundant_placeholder0X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913323___redundant_placeholder1X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913323___redundant_placeholder2X
Tbackward_lstm_93_while_backward_lstm_93_while_cond_55913323___redundant_placeholder3#
backward_lstm_93_while_identity
�
backward_lstm_93/while/LessLess"backward_lstm_93_while_placeholder<backward_lstm_93_while_less_backward_lstm_93_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_93/while/IdentityIdentitybackward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_93_while_identity(backward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_93/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_93/strided_slice_1:

_output_shapes
:
�
�
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_cond_55909573�
|topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_loop_counter�
�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_maximum_iterationsG
Ctopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholderI
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder_1I
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder_2I
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder_3�
~topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_less_topk_bilstm_moe_bidirectional_93_backward_lstm_93_strided_slice_1�
�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_cond_55909573___redundant_placeholder0�
�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_cond_55909573___redundant_placeholder1�
�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_cond_55909573___redundant_placeholder2�
�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_cond_55909573___redundant_placeholder3D
@topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity
�
<topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/LessLessCtopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder~topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_less_topk_bilstm_moe_bidirectional_93_backward_lstm_93_strided_slice_1*
T0*
_output_shapes
: �
@topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/IdentityIdentity@topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Less:z:0*
T0
*
_output_shapes
: "�
@topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identityItopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::| x

_output_shapes
: 
^
_user_specified_nameFDtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/loop_counter:�~

_output_shapes
: 
d
_user_specified_nameLJtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::yu

_output_shapes
: 
[
_user_specified_nameCAtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1:

_output_shapes
:
�	
�
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909948
dense_610_input$
dense_610_55909942: 
dense_610_55909944:
identity��!dense_610/StatefulPartitionedCall�
!dense_610/StatefulPartitionedCallStatefulPartitionedCalldense_610_inputdense_610_55909942dense_610_55909944*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_610_layer_call_and_return_conditional_losses_55909932}
IdentityIdentity*dense_610/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������F
NoOpNoOp"^dense_610/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2F
!dense_610/StatefulPartitionedCall!dense_610/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_610_input:($
"
_user_specified_name
55909942:($
"
_user_specified_name
55909944
�
�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910182

inputs

states
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates:OK
'
_output_shapes
:���������
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
,__inference_lstm_cell_layer_call_fn_55915191

inputs
states_0
states_1
unknown: 
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:���������:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55910182o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1:($
"
_user_specified_name
55915179:($
"
_user_specified_name
55915181:($
"
_user_specified_name
55915183
�
�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915321

inputs
states_0
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:��������� r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:���������V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:���������U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:���������N
ReluRelusplit:output:2*
T0*'
_output_shapes
:���������_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:���������T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:���������K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:���������c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:���������X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:���������Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:���������m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:���������:���������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states_0:QM
'
_output_shapes
:���������
"
_user_specified_name
states_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
r
H__inference_lambda_100_layer_call_and_return_conditional_losses_55911809

inputs
inputs_1
identity�
einsum/EinsumEinsuminputsinputs_1*
N*
T0*&
_output_shapes
:*
equationabc,abd->dabc]
IdentityIdentityeinsum/Einsum:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:::J F
"
_output_shapes
:
 
_user_specified_nameinputs:JF
"
_output_shapes
:
 
_user_specified_nameinputs
�L
�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55915012

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'����������������������������
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55914928*
condR
while_cond_55914927*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
while_cond_55914019
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55914019___redundant_placeholder06
2while_while_cond_55914019___redundant_placeholder16
2while_while_cond_55914019___redundant_placeholder26
2while_while_cond_55914019___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�	
�
while_cond_55911041
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55911041___redundant_placeholder06
2while_while_cond_55911041___redundant_placeholder16
2while_while_cond_55911041___redundant_placeholder26
2while_while_cond_55911041___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
s
G__inference_lambda_99_layer_call_and_return_conditional_losses_55912426
inputs_0
inputs_1
identity~
einsum/EinsumEinsuminputs_0inputs_1*
N*
T0*"
_output_shapes
:*
equationabc,abcd->abdY
IdentityIdentityeinsum/Einsum:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::L H
"
_output_shapes
:
"
_user_specified_name
inputs_0:PL
&
_output_shapes
:
"
_user_specified_name
inputs_1
�	
�
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909720
dense_595_input$
dense_595_55909714: 
dense_595_55909716:
identity��!dense_595/StatefulPartitionedCall�
!dense_595/StatefulPartitionedCallStatefulPartitionedCalldense_595_inputdense_595_55909714dense_595_55909716*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_595_layer_call_and_return_conditional_losses_55909704}
IdentityIdentity*dense_595/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������F
NoOpNoOp"^dense_595/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2F
!dense_595/StatefulPartitionedCall!dense_595/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_595_input:($
"
_user_specified_name
55909714:($
"
_user_specified_name
55909716
�
�
2__inference_topk_bilstm_moe_layer_call_fn_55912198
input_4
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15:	�

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55911782f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:
!
_user_specified_name	input_4:($
"
_user_specified_name
55912160:($
"
_user_specified_name
55912162:($
"
_user_specified_name
55912164:($
"
_user_specified_name
55912166:($
"
_user_specified_name
55912168:($
"
_user_specified_name
55912170:($
"
_user_specified_name
55912172:($
"
_user_specified_name
55912174:(	$
"
_user_specified_name
55912176:(
$
"
_user_specified_name
55912178:($
"
_user_specified_name
55912180:($
"
_user_specified_name
55912182:($
"
_user_specified_name
55912184:($
"
_user_specified_name
55912186:($
"
_user_specified_name
55912188:($
"
_user_specified_name
55912190:($
"
_user_specified_name
55912192:($
"
_user_specified_name
55912194
�r
�
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_body_55909433~
ztopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_loop_counter�
�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_maximum_iterationsF
Btopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholderH
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder_1H
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder_2H
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder_3}
ytopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_strided_slice_1_0�
�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0s
atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: u
ctopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: p
btopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: C
?topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identityE
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_1E
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_2E
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_3E
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_4E
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_5{
wtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_strided_slice_1�
�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_forward_lstm_93_tensorarrayunstack_tensorlistfromtensorq
_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: s
atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: n
`topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��Wtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�Vtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�Xtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
htopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Ztopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0Btopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholderqtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
Vtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpatopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
Gtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMulMatMulatopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Xtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpctopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
Itopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul_1MatMulDtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder_2`topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Dtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/addAddV2Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul:product:0Stopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
Wtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpbtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/BiasAddBiasAddHtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/add:z:0_topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Ptopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/splitSplitYtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/split/split_dim:output:0Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/SigmoidSigmoidOtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/Sigmoid_1SigmoidOtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
Dtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/mulMulNtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/Sigmoid_1:y:0Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
Etopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/ReluReluOtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
Ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/mul_1MulLtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/Sigmoid:y:0Stopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
Ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/add_1AddV2Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/mul:z:0Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/Sigmoid_2SigmoidOtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
Gtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/Relu_1ReluJtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
Ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/mul_2MulNtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/Sigmoid_2:y:0Utopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
[topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemDtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholder_1Btopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholderJtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���~
<topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
:topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/addAddV2Btopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_placeholderEtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: �
>topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
<topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/add_1AddV2ztopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_loop_counterGtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
?topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/IdentityIdentity@topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/add_1:z:0<^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_1Identity�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_maximum_iterations<^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_2Identity>topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/add:z:0<^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_3Identityktopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_4IdentityJtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/mul_2:z:0<^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_5IdentityJtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/add_1:z:0<^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
;topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/NoOpNoOpX^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpW^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpY^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "�
?topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identityHtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity:output:0"�
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_1Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_1:output:0"�
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_2Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_2:output:0"�
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_3Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_3:output:0"�
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_4Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_4:output:0"�
Atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_identity_5Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/Identity_5:output:0"�
`topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourcebtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
atopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourcectopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
_topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_readvariableop_resourceatopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor�topk_bilstm_moe_bidirectional_93_forward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_forward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0"�
wtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_strided_slice_1ytopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_topk_bilstm_moe_bidirectional_93_forward_lstm_93_strided_slice_1_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2�
Wtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpWtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2�
Vtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpVtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2�
Xtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpXtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:{ w

_output_shapes
: 
]
_user_specified_nameECtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/loop_counter:�}

_output_shapes
: 
c
_user_specified_nameKItopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::xt

_output_shapes
: 
Z
_user_specified_nameB@topk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1:��

_output_shapes
: 
r
_user_specified_nameZXtopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
t
H__inference_lambda_100_layer_call_and_return_conditional_losses_55912456
inputs_0
inputs_1
identity�
einsum/EinsumEinsuminputs_0inputs_1*
N*
T0*&
_output_shapes
:*
equationabc,abd->dabc]
IdentityIdentityeinsum/Einsum:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:::L H
"
_output_shapes
:
"
_user_specified_name
inputs_0:LH
"
_output_shapes
:
"
_user_specified_name
inputs_1
�8
�
while_body_55911194
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
r
H__inference_lambda_101_layer_call_and_return_conditional_losses_55911443

inputs
inputs_1
identity|
einsum/EinsumEinsuminputsinputs_1*
N*
T0*"
_output_shapes
:*
equationabcd,ace->acdY
IdentityIdentityeinsum/Einsum:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::N J
&
_output_shapes
:
 
_user_specified_nameinputs:JF
"
_output_shapes
:
 
_user_specified_nameinputs
�
f
-__inference_dropout_93_layer_call_fn_55913705

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dropout_93_layer_call_and_return_conditional_losses_55911757j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
f
H__inference_dropout_93_layer_call_and_return_conditional_losses_55912148

inputs

identity_1I
IdentityIdentityinputs*
T0*"
_output_shapes
:V

Identity_1IdentityIdentity:output:0*
T0*"
_output_shapes
:"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
2__inference_topk_bilstm_moe_layer_call_fn_55912239
input_4
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9: 

unknown_10: 

unknown_11: 

unknown_12: 

unknown_13: 

unknown_14: 

unknown_15:	�

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55912157f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:
!
_user_specified_name	input_4:($
"
_user_specified_name
55912201:($
"
_user_specified_name
55912203:($
"
_user_specified_name
55912205:($
"
_user_specified_name
55912207:($
"
_user_specified_name
55912209:($
"
_user_specified_name
55912211:($
"
_user_specified_name
55912213:($
"
_user_specified_name
55912215:(	$
"
_user_specified_name
55912217:(
$
"
_user_specified_name
55912219:($
"
_user_specified_name
55912221:($
"
_user_specified_name
55912223:($
"
_user_specified_name
55912225:($
"
_user_specified_name
55912227:($
"
_user_specified_name
55912229:($
"
_user_specified_name
55912231:($
"
_user_specified_name
55912233:($
"
_user_specified_name
55912235
�	
�
while_cond_55914782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55914782___redundant_placeholder06
2while_while_cond_55914782___redundant_placeholder16
2while_while_cond_55914782___redundant_placeholder26
2while_while_cond_55914782___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
�
1__inference_sequential_413_layer_call_fn_55909805
dense_600_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_600_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909787s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_600_input:($
"
_user_specified_name
55909799:($
"
_user_specified_name
55909801
�
X
,__inference_lambda_99_layer_call_fn_55912414
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_lambda_99_layer_call_and_return_conditional_losses_55911404[
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::L H
"
_output_shapes
:
"
_user_specified_name
inputs_0:PL
&
_output_shapes
:
"
_user_specified_name
inputs_1
�	
�
while_cond_55914927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55914927___redundant_placeholder06
2while_while_cond_55914927___redundant_placeholder16
2while_while_cond_55914927___redundant_placeholder26
2while_while_cond_55914927___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55910981

inputs*
forward_lstm_93_55910817: *
forward_lstm_93_55910819: &
forward_lstm_93_55910821: +
backward_lstm_93_55910969: +
backward_lstm_93_55910971: '
backward_lstm_93_55910973: 
identity��(backward_lstm_93/StatefulPartitionedCall�'forward_lstm_93/StatefulPartitionedCall�
'forward_lstm_93/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_93_55910817forward_lstm_93_55910819forward_lstm_93_55910821*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55910816�
(backward_lstm_93/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_93_55910969backward_lstm_93_55910971backward_lstm_93_55910973*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55910968X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
	ReverseV2	ReverseV21backward_lstm_93/StatefulPartitionedCall:output:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV20forward_lstm_93/StatefulPartitionedCall:output:0ReverseV2:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :������������������k
IdentityIdentityconcat:output:0^NoOp*
T0*4
_output_shapes"
 :������������������w
NoOpNoOp)^backward_lstm_93/StatefulPartitionedCall(^forward_lstm_93/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2T
(backward_lstm_93/StatefulPartitionedCall(backward_lstm_93/StatefulPartitionedCall2R
'forward_lstm_93/StatefulPartitionedCall'forward_lstm_93/StatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55910817:($
"
_user_specified_name
55910819:($
"
_user_specified_name
55910821:($
"
_user_specified_name
55910969:($
"
_user_specified_name
55910971:($
"
_user_specified_name
55910973
�	
�
while_cond_55910545
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55910545___redundant_placeholder06
2while_while_cond_55910545___redundant_placeholder16
2while_while_cond_55910545___redundant_placeholder26
2while_while_cond_55910545___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�
�
1__inference_sequential_413_layer_call_fn_55909814
dense_600_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_600_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909796s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_600_input:($
"
_user_specified_name
55909808:($
"
_user_specified_name
55909810
�
�
G__inference_dense_594_layer_call_and_return_conditional_losses_55912408

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     p
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Q
SoftmaxSoftmaxBiasAdd:output:0*
T0*"
_output_shapes
:[
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*"
_output_shapes
:V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
��
�
#__inference__wrapped_model_55909671
input_4M
;topk_bilstm_moe_dense_594_tensordot_readvariableop_resource:G
9topk_bilstm_moe_dense_594_biasadd_readvariableop_resource:\
Jtopk_bilstm_moe_sequential_408_dense_595_tensordot_readvariableop_resource:V
Htopk_bilstm_moe_sequential_408_dense_595_biasadd_readvariableop_resource:\
Jtopk_bilstm_moe_sequential_413_dense_600_tensordot_readvariableop_resource:V
Htopk_bilstm_moe_sequential_413_dense_600_biasadd_readvariableop_resource:\
Jtopk_bilstm_moe_sequential_418_dense_605_tensordot_readvariableop_resource:V
Htopk_bilstm_moe_sequential_418_dense_605_biasadd_readvariableop_resource:\
Jtopk_bilstm_moe_sequential_423_dense_610_tensordot_readvariableop_resource:V
Htopk_bilstm_moe_sequential_423_dense_610_biasadd_readvariableop_resource:k
Ytopk_bilstm_moe_bidirectional_93_forward_lstm_93_lstm_cell_matmul_readvariableop_resource: m
[topk_bilstm_moe_bidirectional_93_forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: h
Ztopk_bilstm_moe_bidirectional_93_forward_lstm_93_lstm_cell_biasadd_readvariableop_resource: l
Ztopk_bilstm_moe_bidirectional_93_backward_lstm_93_lstm_cell_matmul_readvariableop_resource: n
\topk_bilstm_moe_bidirectional_93_backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: i
[topk_bilstm_moe_bidirectional_93_backward_lstm_93_lstm_cell_biasadd_readvariableop_resource: K
8topk_bilstm_moe_dense_611_matmul_readvariableop_resource:	�G
9topk_bilstm_moe_dense_611_biasadd_readvariableop_resource:
identity��Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�Qtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul/ReadVariableOp�Stopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�7topk_bilstm_moe/bidirectional_93/backward_lstm_93/while�Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�Ptopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp�Rtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�6topk_bilstm_moe/bidirectional_93/forward_lstm_93/while�0topk_bilstm_moe/dense_594/BiasAdd/ReadVariableOp�2topk_bilstm_moe/dense_594/Tensordot/ReadVariableOp�0topk_bilstm_moe/dense_611/BiasAdd/ReadVariableOp�/topk_bilstm_moe/dense_611/MatMul/ReadVariableOp�?topk_bilstm_moe/sequential_408/dense_595/BiasAdd/ReadVariableOp�Atopk_bilstm_moe/sequential_408/dense_595/Tensordot/ReadVariableOp�?topk_bilstm_moe/sequential_413/dense_600/BiasAdd/ReadVariableOp�Atopk_bilstm_moe/sequential_413/dense_600/Tensordot/ReadVariableOp�?topk_bilstm_moe/sequential_418/dense_605/BiasAdd/ReadVariableOp�Atopk_bilstm_moe/sequential_418/dense_605/Tensordot/ReadVariableOp�?topk_bilstm_moe/sequential_423/dense_610/BiasAdd/ReadVariableOp�Atopk_bilstm_moe/sequential_423/dense_610/Tensordot/ReadVariableOp�
2topk_bilstm_moe/dense_594/Tensordot/ReadVariableOpReadVariableOp;topk_bilstm_moe_dense_594_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
1topk_bilstm_moe/dense_594/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
+topk_bilstm_moe/dense_594/Tensordot/ReshapeReshapeinput_4:topk_bilstm_moe/dense_594/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
*topk_bilstm_moe/dense_594/Tensordot/MatMulMatMul4topk_bilstm_moe/dense_594/Tensordot/Reshape:output:0:topk_bilstm_moe/dense_594/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	�~
)topk_bilstm_moe/dense_594/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
#topk_bilstm_moe/dense_594/TensordotReshape4topk_bilstm_moe/dense_594/Tensordot/MatMul:product:02topk_bilstm_moe/dense_594/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
0topk_bilstm_moe/dense_594/BiasAdd/ReadVariableOpReadVariableOp9topk_bilstm_moe_dense_594_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!topk_bilstm_moe/dense_594/BiasAddBiasAdd,topk_bilstm_moe/dense_594/Tensordot:output:08topk_bilstm_moe/dense_594/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
!topk_bilstm_moe/dense_594/SoftmaxSoftmax*topk_bilstm_moe/dense_594/BiasAdd:output:0*
T0*"
_output_shapes
:j
(topk_bilstm_moe/tf.math.top_k_3/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :�
&topk_bilstm_moe/tf.math.top_k_3/TopKV2TopKV2+topk_bilstm_moe/dense_594/Softmax:softmax:01topk_bilstm_moe/tf.math.top_k_3/TopKV2/k:output:0*
T0*0
_output_shapes
::r
-topk_bilstm_moe/tf.one_hot_3/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?s
.topk_bilstm_moe/tf.one_hot_3/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    l
*topk_bilstm_moe/tf.one_hot_3/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
$topk_bilstm_moe/tf.one_hot_3/one_hotOneHot0topk_bilstm_moe/tf.math.top_k_3/TopKV2:indices:03topk_bilstm_moe/tf.one_hot_3/one_hot/depth:output:06topk_bilstm_moe/tf.one_hot_3/one_hot/on_value:output:07topk_bilstm_moe/tf.one_hot_3/one_hot/off_value:output:0*
T0*
TI0*&
_output_shapes
:�
'topk_bilstm_moe/lambda_99/einsum/EinsumEinsum/topk_bilstm_moe/tf.math.top_k_3/TopKV2:values:0-topk_bilstm_moe/tf.one_hot_3/one_hot:output:0*
N*
T0*"
_output_shapes
:*
equationabc,abcd->abd�
(topk_bilstm_moe/lambda_100/einsum/EinsumEinsuminput_40topk_bilstm_moe/lambda_99/einsum/Einsum:output:0*
N*
T0*&
_output_shapes
:*
equationabc,abd->dabc�
$topk_bilstm_moe/tf.unstack_3/unstackUnpack1topk_bilstm_moe/lambda_100/einsum/Einsum:output:0*
T0*L
_output_shapes:
8::::*	
num�
Atopk_bilstm_moe/sequential_408/dense_595/Tensordot/ReadVariableOpReadVariableOpJtopk_bilstm_moe_sequential_408_dense_595_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
@topk_bilstm_moe/sequential_408/dense_595/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
:topk_bilstm_moe/sequential_408/dense_595/Tensordot/ReshapeReshape-topk_bilstm_moe/tf.unstack_3/unstack:output:0Itopk_bilstm_moe/sequential_408/dense_595/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
9topk_bilstm_moe/sequential_408/dense_595/Tensordot/MatMulMatMulCtopk_bilstm_moe/sequential_408/dense_595/Tensordot/Reshape:output:0Itopk_bilstm_moe/sequential_408/dense_595/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
8topk_bilstm_moe/sequential_408/dense_595/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
2topk_bilstm_moe/sequential_408/dense_595/TensordotReshapeCtopk_bilstm_moe/sequential_408/dense_595/Tensordot/MatMul:product:0Atopk_bilstm_moe/sequential_408/dense_595/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
?topk_bilstm_moe/sequential_408/dense_595/BiasAdd/ReadVariableOpReadVariableOpHtopk_bilstm_moe_sequential_408_dense_595_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
0topk_bilstm_moe/sequential_408/dense_595/BiasAddBiasAdd;topk_bilstm_moe/sequential_408/dense_595/Tensordot:output:0Gtopk_bilstm_moe/sequential_408/dense_595/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
-topk_bilstm_moe/sequential_408/dense_595/ReluRelu9topk_bilstm_moe/sequential_408/dense_595/BiasAdd:output:0*
T0*"
_output_shapes
:�
Atopk_bilstm_moe/sequential_413/dense_600/Tensordot/ReadVariableOpReadVariableOpJtopk_bilstm_moe_sequential_413_dense_600_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
@topk_bilstm_moe/sequential_413/dense_600/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
:topk_bilstm_moe/sequential_413/dense_600/Tensordot/ReshapeReshape-topk_bilstm_moe/tf.unstack_3/unstack:output:1Itopk_bilstm_moe/sequential_413/dense_600/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
9topk_bilstm_moe/sequential_413/dense_600/Tensordot/MatMulMatMulCtopk_bilstm_moe/sequential_413/dense_600/Tensordot/Reshape:output:0Itopk_bilstm_moe/sequential_413/dense_600/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
8topk_bilstm_moe/sequential_413/dense_600/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
2topk_bilstm_moe/sequential_413/dense_600/TensordotReshapeCtopk_bilstm_moe/sequential_413/dense_600/Tensordot/MatMul:product:0Atopk_bilstm_moe/sequential_413/dense_600/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
?topk_bilstm_moe/sequential_413/dense_600/BiasAdd/ReadVariableOpReadVariableOpHtopk_bilstm_moe_sequential_413_dense_600_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
0topk_bilstm_moe/sequential_413/dense_600/BiasAddBiasAdd;topk_bilstm_moe/sequential_413/dense_600/Tensordot:output:0Gtopk_bilstm_moe/sequential_413/dense_600/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
-topk_bilstm_moe/sequential_413/dense_600/ReluRelu9topk_bilstm_moe/sequential_413/dense_600/BiasAdd:output:0*
T0*"
_output_shapes
:�
Atopk_bilstm_moe/sequential_418/dense_605/Tensordot/ReadVariableOpReadVariableOpJtopk_bilstm_moe_sequential_418_dense_605_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
@topk_bilstm_moe/sequential_418/dense_605/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
:topk_bilstm_moe/sequential_418/dense_605/Tensordot/ReshapeReshape-topk_bilstm_moe/tf.unstack_3/unstack:output:2Itopk_bilstm_moe/sequential_418/dense_605/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
9topk_bilstm_moe/sequential_418/dense_605/Tensordot/MatMulMatMulCtopk_bilstm_moe/sequential_418/dense_605/Tensordot/Reshape:output:0Itopk_bilstm_moe/sequential_418/dense_605/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
8topk_bilstm_moe/sequential_418/dense_605/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
2topk_bilstm_moe/sequential_418/dense_605/TensordotReshapeCtopk_bilstm_moe/sequential_418/dense_605/Tensordot/MatMul:product:0Atopk_bilstm_moe/sequential_418/dense_605/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
?topk_bilstm_moe/sequential_418/dense_605/BiasAdd/ReadVariableOpReadVariableOpHtopk_bilstm_moe_sequential_418_dense_605_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
0topk_bilstm_moe/sequential_418/dense_605/BiasAddBiasAdd;topk_bilstm_moe/sequential_418/dense_605/Tensordot:output:0Gtopk_bilstm_moe/sequential_418/dense_605/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
-topk_bilstm_moe/sequential_418/dense_605/ReluRelu9topk_bilstm_moe/sequential_418/dense_605/BiasAdd:output:0*
T0*"
_output_shapes
:�
Atopk_bilstm_moe/sequential_423/dense_610/Tensordot/ReadVariableOpReadVariableOpJtopk_bilstm_moe_sequential_423_dense_610_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
@topk_bilstm_moe/sequential_423/dense_610/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
:topk_bilstm_moe/sequential_423/dense_610/Tensordot/ReshapeReshape-topk_bilstm_moe/tf.unstack_3/unstack:output:3Itopk_bilstm_moe/sequential_423/dense_610/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
9topk_bilstm_moe/sequential_423/dense_610/Tensordot/MatMulMatMulCtopk_bilstm_moe/sequential_423/dense_610/Tensordot/Reshape:output:0Itopk_bilstm_moe/sequential_423/dense_610/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
8topk_bilstm_moe/sequential_423/dense_610/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
2topk_bilstm_moe/sequential_423/dense_610/TensordotReshapeCtopk_bilstm_moe/sequential_423/dense_610/Tensordot/MatMul:product:0Atopk_bilstm_moe/sequential_423/dense_610/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
?topk_bilstm_moe/sequential_423/dense_610/BiasAdd/ReadVariableOpReadVariableOpHtopk_bilstm_moe_sequential_423_dense_610_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
0topk_bilstm_moe/sequential_423/dense_610/BiasAddBiasAdd;topk_bilstm_moe/sequential_423/dense_610/Tensordot:output:0Gtopk_bilstm_moe/sequential_423/dense_610/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
-topk_bilstm_moe/sequential_423/dense_610/ReluRelu9topk_bilstm_moe/sequential_423/dense_610/BiasAdd:output:0*
T0*"
_output_shapes
:�
!topk_bilstm_moe/tf.stack_94/stackPack;topk_bilstm_moe/sequential_408/dense_595/Relu:activations:0;topk_bilstm_moe/sequential_413/dense_600/Relu:activations:0;topk_bilstm_moe/sequential_418/dense_605/Relu:activations:0;topk_bilstm_moe/sequential_423/dense_610/Relu:activations:0*
N*
T0*&
_output_shapes
:*

axis�
(topk_bilstm_moe/lambda_101/einsum/EinsumEinsum*topk_bilstm_moe/tf.stack_94/stack:output:00topk_bilstm_moe/lambda_99/einsum/Einsum:output:0*
N*
T0*"
_output_shapes
:*
equationabcd,ace->acd�
6topk_bilstm_moe/bidirectional_93/forward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
Dtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
>topk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_sliceStridedSlice?topk_bilstm_moe/bidirectional_93/forward_lstm_93/Shape:output:0Mtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice/stack:output:0Otopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice/stack_1:output:0Otopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
?topk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
=topk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros/packedPackGtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice:output:0Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:�
<topk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
6topk_bilstm_moe/bidirectional_93/forward_lstm_93/zerosFillFtopk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros/packed:output:0Etopk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:�
Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
?topk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros_1/packedPackGtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice:output:0Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:�
>topk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
8topk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros_1FillHtopk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros_1/packed:output:0Gtopk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:�
?topk_bilstm_moe/bidirectional_93/forward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
:topk_bilstm_moe/bidirectional_93/forward_lstm_93/transpose	Transpose1topk_bilstm_moe/lambda_101/einsum/Einsum:output:0Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:�
8topk_bilstm_moe/bidirectional_93/forward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         �
Ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
@topk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1StridedSliceAtopk_bilstm_moe/bidirectional_93/forward_lstm_93/Shape_1:output:0Otopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1/stack:output:0Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1/stack_1:output:0Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Ltopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
>topk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2TensorListReserveUtopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2/element_shape:output:0Itopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Xtopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor>topk_bilstm_moe/bidirectional_93/forward_lstm_93/transpose:y:0otopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
@topk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_2StridedSlice>topk_bilstm_moe/bidirectional_93/forward_lstm_93/transpose:y:0Otopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_2/stack:output:0Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_2/stack_1:output:0Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
Ptopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOpYtopk_bilstm_moe_bidirectional_93_forward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMulMatMulItopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_2:output:0Xtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Rtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp[topk_bilstm_moe_bidirectional_93_forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
Ctopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul_1MatMul?topk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros:output:0Ztopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
>topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/addAddV2Ktopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul:product:0Mtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpZtopk_bilstm_moe_bidirectional_93_forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Btopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/BiasAddBiasAddBtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/add:z:0Ytopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
@topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/splitSplitStopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/split/split_dim:output:0Ktopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
Btopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/SigmoidSigmoidItopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
Dtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/Sigmoid_1SigmoidItopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
>topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/mulMulHtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/Sigmoid_1:y:0Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:�
?topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/ReluReluItopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
@topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/mul_1MulFtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/Sigmoid:y:0Mtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
@topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/add_1AddV2Btopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/mul:z:0Dtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
Dtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/Sigmoid_2SigmoidItopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:�
Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/Relu_1ReluDtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
@topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/mul_2MulHtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/Sigmoid_2:y:0Otopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
Ntopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
@topk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2_1TensorListReserveWtopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2_1/element_shape:output:0Itopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���w
5topk_bilstm_moe/bidirectional_93/forward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : �
Itopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
����������
Ctopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

6topk_bilstm_moe/bidirectional_93/forward_lstm_93/whileWhileLtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/loop_counter:output:0Rtopk_bilstm_moe/bidirectional_93/forward_lstm_93/while/maximum_iterations:output:0>topk_bilstm_moe/bidirectional_93/forward_lstm_93/time:output:0Itopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2_1:handle:0?topk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros:output:0Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/zeros_1:output:0Itopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_1:output:0htopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ytopk_bilstm_moe_bidirectional_93_forward_lstm_93_lstm_cell_matmul_readvariableop_resource[topk_bilstm_moe_bidirectional_93_forward_lstm_93_lstm_cell_matmul_1_readvariableop_resourceZtopk_bilstm_moe_bidirectional_93_forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*P
bodyHRF
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_body_55909433*P
condHRF
Dtopk_bilstm_moe_bidirectional_93_forward_lstm_93_while_cond_55909432*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
atopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Stopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStack?topk_bilstm_moe/bidirectional_93/forward_lstm_93/while:output:3jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0�
Ftopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Htopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
@topk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_3StridedSlice\topk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0Otopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_3/stack:output:0Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_3/stack_1:output:0Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
Atopk_bilstm_moe/bidirectional_93/forward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
<topk_bilstm_moe/bidirectional_93/forward_lstm_93/transpose_1	Transpose\topk_bilstm_moe/bidirectional_93/forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0Jtopk_bilstm_moe/bidirectional_93/forward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:�
8topk_bilstm_moe/bidirectional_93/forward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    �
7topk_bilstm_moe/bidirectional_93/backward_lstm_93/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
Etopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
?topk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_sliceStridedSlice@topk_bilstm_moe/bidirectional_93/backward_lstm_93/Shape:output:0Ntopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice/stack:output:0Ptopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice/stack_1:output:0Ptopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
@topk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
>topk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros/packedPackHtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice:output:0Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:�
=topk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
7topk_bilstm_moe/bidirectional_93/backward_lstm_93/zerosFillGtopk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros/packed:output:0Ftopk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros/Const:output:0*
T0*
_output_shapes

:�
Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
@topk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros_1/packedPackHtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice:output:0Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:�
?topk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
9topk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros_1FillItopk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros_1/packed:output:0Htopk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros_1/Const:output:0*
T0*
_output_shapes

:�
@topk_bilstm_moe/bidirectional_93/backward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
;topk_bilstm_moe/bidirectional_93/backward_lstm_93/transpose	Transpose1topk_bilstm_moe/lambda_101/einsum/Einsum:output:0Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/transpose/perm:output:0*
T0*"
_output_shapes
:�
9topk_bilstm_moe/bidirectional_93/backward_lstm_93/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         �
Gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Atopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1StridedSliceBtopk_bilstm_moe/bidirectional_93/backward_lstm_93/Shape_1:output:0Ptopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1/stack:output:0Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1/stack_1:output:0Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
Mtopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
?topk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2TensorListReserveVtopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2/element_shape:output:0Jtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
@topk_bilstm_moe/bidirectional_93/backward_lstm_93/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
;topk_bilstm_moe/bidirectional_93/backward_lstm_93/ReverseV2	ReverseV2?topk_bilstm_moe/bidirectional_93/backward_lstm_93/transpose:y:0Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/ReverseV2/axis:output:0*
T0*"
_output_shapes
:�
gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Ytopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorDtopk_bilstm_moe/bidirectional_93/backward_lstm_93/ReverseV2:output:0ptopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Atopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_2StridedSlice?topk_bilstm_moe/bidirectional_93/backward_lstm_93/transpose:y:0Ptopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_2/stack:output:0Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_2/stack_1:output:0Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
Qtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOpZtopk_bilstm_moe_bidirectional_93_backward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMulMatMulJtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_2:output:0Ytopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Stopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp\topk_bilstm_moe_bidirectional_93_backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
Dtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul_1MatMul@topk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros:output:0[topk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
?topk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/addAddV2Ltopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul:product:0Ntopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp[topk_bilstm_moe_bidirectional_93_backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
Ctopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/BiasAddBiasAddCtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/add:z:0Ztopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Atopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/splitSplitTtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/split/split_dim:output:0Ltopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
Ctopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/SigmoidSigmoidJtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/split:output:0*
T0*
_output_shapes

:�
Etopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/Sigmoid_1SigmoidJtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/split:output:1*
T0*
_output_shapes

:�
?topk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/mulMulItopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/Sigmoid_1:y:0Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros_1:output:0*
T0*
_output_shapes

:�
@topk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/ReluReluJtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/split:output:2*
T0*
_output_shapes

:�
Atopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/mul_1MulGtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/Sigmoid:y:0Ntopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
Atopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/add_1AddV2Ctopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/mul:z:0Etopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
Etopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/Sigmoid_2SigmoidJtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/split:output:3*
T0*
_output_shapes

:�
Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/Relu_1ReluEtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
Atopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/mul_2MulItopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/Sigmoid_2:y:0Ptopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
Otopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Atopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2_1TensorListReserveXtopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2_1/element_shape:output:0Jtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���x
6topk_bilstm_moe/bidirectional_93/backward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : �
Jtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
����������
Dtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �

7topk_bilstm_moe/bidirectional_93/backward_lstm_93/whileWhileMtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/loop_counter:output:0Stopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/maximum_iterations:output:0?topk_bilstm_moe/bidirectional_93/backward_lstm_93/time:output:0Jtopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2_1:handle:0@topk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros:output:0Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/zeros_1:output:0Jtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1:output:0itopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ztopk_bilstm_moe_bidirectional_93_backward_lstm_93_lstm_cell_matmul_readvariableop_resource\topk_bilstm_moe_bidirectional_93_backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource[topk_bilstm_moe_bidirectional_93_backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*:
_output_shapes(
&: : : : ::: : : : : *%
_read_only_resource_inputs
	
*Q
bodyIRG
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_body_55909574*Q
condIRG
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_cond_55909573*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations �
btopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
Ttopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStack@topk_bilstm_moe/bidirectional_93/backward_lstm_93/while:output:3ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0�
Gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
����������
Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: �
Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Atopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_3StridedSlice]topk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0Ptopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_3/stack:output:0Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_3/stack_1:output:0Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask�
Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
=topk_bilstm_moe/bidirectional_93/backward_lstm_93/transpose_1	Transpose]topk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/transpose_1/perm:output:0*
T0*"
_output_shapes
:�
9topk_bilstm_moe/bidirectional_93/backward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    y
/topk_bilstm_moe/bidirectional_93/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
*topk_bilstm_moe/bidirectional_93/ReverseV2	ReverseV2Atopk_bilstm_moe/bidirectional_93/backward_lstm_93/transpose_1:y:08topk_bilstm_moe/bidirectional_93/ReverseV2/axis:output:0*
T0*"
_output_shapes
:n
,topk_bilstm_moe/bidirectional_93/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
'topk_bilstm_moe/bidirectional_93/concatConcatV2@topk_bilstm_moe/bidirectional_93/forward_lstm_93/transpose_1:y:03topk_bilstm_moe/bidirectional_93/ReverseV2:output:05topk_bilstm_moe/bidirectional_93/concat/axis:output:0*
N*
T0*"
_output_shapes
:�
#topk_bilstm_moe/dropout_93/IdentityIdentity0topk_bilstm_moe/bidirectional_93/concat:output:0*
T0*"
_output_shapes
:q
 topk_bilstm_moe/flatten_93/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
"topk_bilstm_moe/flatten_93/ReshapeReshape,topk_bilstm_moe/dropout_93/Identity:output:0)topk_bilstm_moe/flatten_93/Const:output:0*
T0*
_output_shapes
:	��
/topk_bilstm_moe/dense_611/MatMul/ReadVariableOpReadVariableOp8topk_bilstm_moe_dense_611_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
 topk_bilstm_moe/dense_611/MatMulMatMul+topk_bilstm_moe/flatten_93/Reshape:output:07topk_bilstm_moe/dense_611/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
0topk_bilstm_moe/dense_611/BiasAdd/ReadVariableOpReadVariableOp9topk_bilstm_moe_dense_611_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!topk_bilstm_moe/dense_611/BiasAddBiasAdd*topk_bilstm_moe/dense_611/MatMul:product:08topk_bilstm_moe/dense_611/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:p
IdentityIdentity*topk_bilstm_moe/dense_611/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:�

NoOpNoOpS^topk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpR^topk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul/ReadVariableOpT^topk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp8^topk_bilstm_moe/bidirectional_93/backward_lstm_93/whileR^topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpQ^topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul/ReadVariableOpS^topk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp7^topk_bilstm_moe/bidirectional_93/forward_lstm_93/while1^topk_bilstm_moe/dense_594/BiasAdd/ReadVariableOp3^topk_bilstm_moe/dense_594/Tensordot/ReadVariableOp1^topk_bilstm_moe/dense_611/BiasAdd/ReadVariableOp0^topk_bilstm_moe/dense_611/MatMul/ReadVariableOp@^topk_bilstm_moe/sequential_408/dense_595/BiasAdd/ReadVariableOpB^topk_bilstm_moe/sequential_408/dense_595/Tensordot/ReadVariableOp@^topk_bilstm_moe/sequential_413/dense_600/BiasAdd/ReadVariableOpB^topk_bilstm_moe/sequential_413/dense_600/Tensordot/ReadVariableOp@^topk_bilstm_moe/sequential_418/dense_605/BiasAdd/ReadVariableOpB^topk_bilstm_moe/sequential_418/dense_605/Tensordot/ReadVariableOp@^topk_bilstm_moe/sequential_423/dense_610/BiasAdd/ReadVariableOpB^topk_bilstm_moe/sequential_423/dense_610/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 2�
Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpRtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2�
Qtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul/ReadVariableOpQtopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul/ReadVariableOp2�
Stopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpStopk_bilstm_moe/bidirectional_93/backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2r
7topk_bilstm_moe/bidirectional_93/backward_lstm_93/while7topk_bilstm_moe/bidirectional_93/backward_lstm_93/while2�
Qtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpQtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2�
Ptopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul/ReadVariableOpPtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2�
Rtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpRtopk_bilstm_moe/bidirectional_93/forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2p
6topk_bilstm_moe/bidirectional_93/forward_lstm_93/while6topk_bilstm_moe/bidirectional_93/forward_lstm_93/while2d
0topk_bilstm_moe/dense_594/BiasAdd/ReadVariableOp0topk_bilstm_moe/dense_594/BiasAdd/ReadVariableOp2h
2topk_bilstm_moe/dense_594/Tensordot/ReadVariableOp2topk_bilstm_moe/dense_594/Tensordot/ReadVariableOp2d
0topk_bilstm_moe/dense_611/BiasAdd/ReadVariableOp0topk_bilstm_moe/dense_611/BiasAdd/ReadVariableOp2b
/topk_bilstm_moe/dense_611/MatMul/ReadVariableOp/topk_bilstm_moe/dense_611/MatMul/ReadVariableOp2�
?topk_bilstm_moe/sequential_408/dense_595/BiasAdd/ReadVariableOp?topk_bilstm_moe/sequential_408/dense_595/BiasAdd/ReadVariableOp2�
Atopk_bilstm_moe/sequential_408/dense_595/Tensordot/ReadVariableOpAtopk_bilstm_moe/sequential_408/dense_595/Tensordot/ReadVariableOp2�
?topk_bilstm_moe/sequential_413/dense_600/BiasAdd/ReadVariableOp?topk_bilstm_moe/sequential_413/dense_600/BiasAdd/ReadVariableOp2�
Atopk_bilstm_moe/sequential_413/dense_600/Tensordot/ReadVariableOpAtopk_bilstm_moe/sequential_413/dense_600/Tensordot/ReadVariableOp2�
?topk_bilstm_moe/sequential_418/dense_605/BiasAdd/ReadVariableOp?topk_bilstm_moe/sequential_418/dense_605/BiasAdd/ReadVariableOp2�
Atopk_bilstm_moe/sequential_418/dense_605/Tensordot/ReadVariableOpAtopk_bilstm_moe/sequential_418/dense_605/Tensordot/ReadVariableOp2�
?topk_bilstm_moe/sequential_423/dense_610/BiasAdd/ReadVariableOp?topk_bilstm_moe/sequential_423/dense_610/BiasAdd/ReadVariableOp2�
Atopk_bilstm_moe/sequential_423/dense_610/Tensordot/ReadVariableOpAtopk_bilstm_moe/sequential_423/dense_610/Tensordot/ReadVariableOp:K G
"
_output_shapes
:
!
_user_specified_name	input_4:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
Y
-__inference_lambda_101_layer_call_fn_55912468
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lambda_101_layer_call_and_return_conditional_losses_55911841[
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::P L
&
_output_shapes
:
"
_user_specified_name
inputs_0:LH
"
_output_shapes
:
"
_user_specified_name
inputs_1
�	
�
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909872
dense_605_input$
dense_605_55909866: 
dense_605_55909868:
identity��!dense_605/StatefulPartitionedCall�
!dense_605/StatefulPartitionedCallStatefulPartitionedCalldense_605_inputdense_605_55909866dense_605_55909868*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_605_layer_call_and_return_conditional_losses_55909856}
IdentityIdentity*dense_605/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������F
NoOpNoOp"^dense_605/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2F
!dense_605/StatefulPartitionedCall!dense_605/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_605_input:($
"
_user_specified_name
55909866:($
"
_user_specified_name
55909868
�

�
3__inference_bidirectional_93_layer_call_fn_55912548

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55912130j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
55912534:($
"
_user_specified_name
55912536:($
"
_user_specified_name
55912538:($
"
_user_specified_name
55912540:($
"
_user_specified_name
55912542:($
"
_user_specified_name
55912544
�J
�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914247
inputs_0:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55914163*
condR
while_cond_55914162*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
Y
-__inference_lambda_100_layer_call_fn_55912438
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_lambda_100_layer_call_and_return_conditional_losses_55911411_
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:::L H
"
_output_shapes
:
"
_user_specified_name
inputs_0:LH
"
_output_shapes
:
"
_user_specified_name
inputs_1
�s
�
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_body_55909574�
|topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_loop_counter�
�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_maximum_iterationsG
Ctopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholderI
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder_1I
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder_2I
Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder_3
{topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_strided_slice_1_0�
�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0t
btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0: v
dtopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0: q
ctopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0: D
@topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identityF
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_1F
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_2F
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_3F
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_4F
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_5}
ytopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_strided_slice_1�
�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_backward_lstm_93_tensorarrayunstack_tensorlistfromtensorr
`topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource: t
btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource: o
atopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource: ��Xtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp�Wtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp�Ytopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp�
itopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
[topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0Ctopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholderrtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0�
Wtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpbtopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
Htopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMulMatMulbtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/TensorArrayV2Read/TensorListGetItem:item:0_topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Ytopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpdtopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
Jtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul_1MatMulEtopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder_2atopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Etopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/addAddV2Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul:product:0Ttopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: �
Xtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpctopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/BiasAddBiasAddItopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/add:z:0`topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: �
Qtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
Gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/splitSplitZtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/split/split_dim:output:0Rtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split�
Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/SigmoidSigmoidPtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/split:output:0*
T0*
_output_shapes

:�
Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/Sigmoid_1SigmoidPtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/split:output:1*
T0*
_output_shapes

:�
Etopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/mulMulOtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/Sigmoid_1:y:0Etopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder_3*
T0*
_output_shapes

:�
Ftopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/ReluReluPtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/split:output:2*
T0*
_output_shapes

:�
Gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/mul_1MulMtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/Sigmoid:y:0Ttopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:�
Gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/add_1AddV2Itopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/mul:z:0Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:�
Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/Sigmoid_2SigmoidPtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/split:output:3*
T0*
_output_shapes

:�
Htopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/Relu_1ReluKtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:�
Gtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/mul_2MulOtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/Sigmoid_2:y:0Vtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:�
\topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemEtopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholder_1Ctopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholderKtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���
=topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
;topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/addAddV2Ctopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_placeholderFtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/add/y:output:0*
T0*
_output_shapes
: �
?topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
=topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/add_1AddV2|topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_loop_counterHtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/add_1/y:output:0*
T0*
_output_shapes
: �
@topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/IdentityIdentityAtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/add_1:z:0=^topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_1Identity�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_maximum_iterations=^topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_2Identity?topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/add:z:0=^topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_3Identityltopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/NoOp*
T0*
_output_shapes
: �
Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_4IdentityKtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/mul_2:z:0=^topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
Btopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_5IdentityKtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/add_1:z:0=^topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/NoOp*
T0*
_output_shapes

:�
<topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/NoOpNoOpY^topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpX^topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpZ^topk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "�
@topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identityItopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity:output:0"�
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_1Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_1:output:0"�
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_2Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_2:output:0"�
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_3Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_3:output:0"�
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_4Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_4:output:0"�
Btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_identity_5Ktopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/Identity_5:output:0"�
atopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resourcectopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_biasadd_readvariableop_resource_0"�
btopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resourcedtopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_1_readvariableop_resource_0"�
`topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_readvariableop_resourcebtopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_lstm_cell_matmul_readvariableop_resource_0"�
�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor�topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_93_backward_lstm_93_tensorarrayunstack_tensorlistfromtensor_0"�
ytopk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_strided_slice_1{topk_bilstm_moe_bidirectional_93_backward_lstm_93_while_topk_bilstm_moe_bidirectional_93_backward_lstm_93_strided_slice_1_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2�
Xtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOpXtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/BiasAdd/ReadVariableOp2�
Wtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOpWtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul/ReadVariableOp2�
Ytopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOpYtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/lstm_cell/MatMul_1/ReadVariableOp:| x

_output_shapes
: 
^
_user_specified_nameFDtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/loop_counter:�~

_output_shapes
: 
d
_user_specified_nameLJtopk_bilstm_moe/bidirectional_93/backward_lstm_93/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::yu

_output_shapes
: 
[
_user_specified_nameCAtopk_bilstm_moe/bidirectional_93/backward_lstm_93/strided_slice_1:��

_output_shapes
: 
s
_user_specified_name[Ytopk_bilstm_moe/bidirectional_93/backward_lstm_93/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�	
�
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909939
dense_610_input$
dense_610_55909933: 
dense_610_55909935:
identity��!dense_610/StatefulPartitionedCall�
!dense_610/StatefulPartitionedCallStatefulPartitionedCalldense_610_inputdense_610_55909933dense_610_55909935*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_610_layer_call_and_return_conditional_losses_55909932}
IdentityIdentity*dense_610/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������F
NoOpNoOp"^dense_610/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2F
!dense_610/StatefulPartitionedCall!dense_610/StatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_610_input:($
"
_user_specified_name
55909933:($
"
_user_specified_name
55909935
�

�
3__inference_bidirectional_93_layer_call_fn_55912531

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55911732j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
55912517:($
"
_user_specified_name
55912519:($
"
_user_specified_name
55912521:($
"
_user_specified_name
55912523:($
"
_user_specified_name
55912525:($
"
_user_specified_name
55912527
�
�
G__inference_dense_600_layer_call_and_return_conditional_losses_55913837

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
2__inference_forward_lstm_93_layer_call_fn_55913928
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55910120|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55913920:($
"
_user_specified_name
55913922:($
"
_user_specified_name
55913924
��
�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913124
inputs_0J
8forward_lstm_93_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_93_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource: 
identity��1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp�2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�backward_lstm_93/while�0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp�/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp�1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp�forward_lstm_93/while[
forward_lstm_93/ShapeShapeinputs_0*
T0*
_output_shapes
::��m
#forward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_sliceStridedSliceforward_lstm_93/Shape:output:0,forward_lstm_93/strided_slice/stack:output:0.forward_lstm_93/strided_slice/stack_1:output:0.forward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros/packedPack&forward_lstm_93/strided_slice:output:0'forward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zerosFill%forward_lstm_93/zeros/packed:output:0$forward_lstm_93/zeros/Const:output:0*
T0*'
_output_shapes
:���������b
 forward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/zeros_1/packedPack&forward_lstm_93/strided_slice:output:0)forward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
forward_lstm_93/zeros_1Fill'forward_lstm_93/zeros_1/packed:output:0&forward_lstm_93/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������s
forward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose	Transposeinputs_0'forward_lstm_93/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������r
forward_lstm_93/Shape_1Shapeforward_lstm_93/transpose:y:0*
T0*
_output_shapes
::��o
%forward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_1StridedSlice forward_lstm_93/Shape_1:output:0.forward_lstm_93/strided_slice_1/stack:output:00forward_lstm_93/strided_slice_1/stack_1:output:00forward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
forward_lstm_93/TensorArrayV2TensorListReserve4forward_lstm_93/TensorArrayV2/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Eforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
7forward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_93/transpose:y:0Nforward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���o
%forward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_2StridedSliceforward_lstm_93/transpose:y:0.forward_lstm_93/strided_slice_2/stack:output:00forward_lstm_93/strided_slice_2/stack_1:output:00forward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
 forward_lstm_93/lstm_cell/MatMulMatMul(forward_lstm_93/strided_slice_2:output:07forward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
"forward_lstm_93/lstm_cell/MatMul_1MatMulforward_lstm_93/zeros:output:09forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
forward_lstm_93/lstm_cell/addAddV2*forward_lstm_93/lstm_cell/MatMul:product:0,forward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
!forward_lstm_93/lstm_cell/BiasAddBiasAdd!forward_lstm_93/lstm_cell/add:z:08forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� k
)forward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
forward_lstm_93/lstm_cell/splitSplit2forward_lstm_93/lstm_cell/split/split_dim:output:0*forward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
!forward_lstm_93/lstm_cell/SigmoidSigmoid(forward_lstm_93/lstm_cell/split:output:0*
T0*'
_output_shapes
:����������
#forward_lstm_93/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_93/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/mulMul'forward_lstm_93/lstm_cell/Sigmoid_1:y:0 forward_lstm_93/zeros_1:output:0*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/ReluRelu(forward_lstm_93/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/mul_1Mul%forward_lstm_93/lstm_cell/Sigmoid:y:0,forward_lstm_93/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/add_1AddV2!forward_lstm_93/lstm_cell/mul:z:0#forward_lstm_93/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:����������
#forward_lstm_93/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_93/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������
 forward_lstm_93/lstm_cell/Relu_1Relu#forward_lstm_93/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
forward_lstm_93/lstm_cell/mul_2Mul'forward_lstm_93/lstm_cell/Sigmoid_2:y:0.forward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������~
-forward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
forward_lstm_93/TensorArrayV2_1TensorListReserve6forward_lstm_93/TensorArrayV2_1/element_shape:output:0(forward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���V
forward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������d
"forward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
forward_lstm_93/whileWhile+forward_lstm_93/while/loop_counter:output:01forward_lstm_93/while/maximum_iterations:output:0forward_lstm_93/time:output:0(forward_lstm_93/TensorArrayV2_1:handle:0forward_lstm_93/zeros:output:0 forward_lstm_93/zeros_1:output:0(forward_lstm_93/strided_slice_1:output:0Gforward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_93_lstm_cell_matmul_readvariableop_resource:forward_lstm_93_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#forward_lstm_93_while_body_55912895*/
cond'R%
#forward_lstm_93_while_cond_55912894*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
@forward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
2forward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_93/while:output:3Iforward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0x
%forward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������q
'forward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
forward_lstm_93/strided_slice_3StridedSlice;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_93/strided_slice_3/stack:output:00forward_lstm_93/strided_slice_3/stack_1:output:00forward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_masku
 forward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
forward_lstm_93/transpose_1	Transpose;forward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_93/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
forward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
backward_lstm_93/ShapeShapeinputs_0*
T0*
_output_shapes
::��n
$backward_lstm_93/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_93/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_93/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
backward_lstm_93/strided_sliceStridedSlicebackward_lstm_93/Shape:output:0-backward_lstm_93/strided_slice/stack:output:0/backward_lstm_93/strided_slice/stack_1:output:0/backward_lstm_93/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_93/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros/packedPack'backward_lstm_93/strided_slice:output:0(backward_lstm_93/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_93/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zerosFill&backward_lstm_93/zeros/packed:output:0%backward_lstm_93/zeros/Const:output:0*
T0*'
_output_shapes
:���������c
!backward_lstm_93/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
backward_lstm_93/zeros_1/packedPack'backward_lstm_93/strided_slice:output:0*backward_lstm_93/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_93/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
backward_lstm_93/zeros_1Fill(backward_lstm_93/zeros_1/packed:output:0'backward_lstm_93/zeros_1/Const:output:0*
T0*'
_output_shapes
:���������t
backward_lstm_93/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose	Transposeinputs_0(backward_lstm_93/transpose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������t
backward_lstm_93/Shape_1Shapebackward_lstm_93/transpose:y:0*
T0*
_output_shapes
::��p
&backward_lstm_93/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_1StridedSlice!backward_lstm_93/Shape_1:output:0/backward_lstm_93/strided_slice_1/stack:output:01backward_lstm_93/strided_slice_1/stack_1:output:01backward_lstm_93/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_93/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
backward_lstm_93/TensorArrayV2TensorListReserve5backward_lstm_93/TensorArrayV2/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���i
backward_lstm_93/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
backward_lstm_93/ReverseV2	ReverseV2backward_lstm_93/transpose:y:0(backward_lstm_93/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'����������������������������
Fbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
8backward_lstm_93/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_93/ReverseV2:output:0Obackward_lstm_93/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���p
&backward_lstm_93/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_93/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_2StridedSlicebackward_lstm_93/transpose:y:0/backward_lstm_93/strided_slice_2/stack:output:01backward_lstm_93/strided_slice_2/stack_1:output:01backward_lstm_93/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_93_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
!backward_lstm_93/lstm_cell/MatMulMatMul)backward_lstm_93/strided_slice_2:output:08backward_lstm_93/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
#backward_lstm_93/lstm_cell/MatMul_1MatMulbackward_lstm_93/zeros:output:0:backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
backward_lstm_93/lstm_cell/addAddV2+backward_lstm_93/lstm_cell/MatMul:product:0-backward_lstm_93/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
"backward_lstm_93/lstm_cell/BiasAddBiasAdd"backward_lstm_93/lstm_cell/add:z:09backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� l
*backward_lstm_93/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
 backward_lstm_93/lstm_cell/splitSplit3backward_lstm_93/lstm_cell/split/split_dim:output:0+backward_lstm_93/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_split�
"backward_lstm_93/lstm_cell/SigmoidSigmoid)backward_lstm_93/lstm_cell/split:output:0*
T0*'
_output_shapes
:����������
$backward_lstm_93/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_93/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
backward_lstm_93/lstm_cell/mulMul(backward_lstm_93/lstm_cell/Sigmoid_1:y:0!backward_lstm_93/zeros_1:output:0*
T0*'
_output_shapes
:����������
backward_lstm_93/lstm_cell/ReluRelu)backward_lstm_93/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
 backward_lstm_93/lstm_cell/mul_1Mul&backward_lstm_93/lstm_cell/Sigmoid:y:0-backward_lstm_93/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
 backward_lstm_93/lstm_cell/add_1AddV2"backward_lstm_93/lstm_cell/mul:z:0$backward_lstm_93/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:����������
$backward_lstm_93/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_93/lstm_cell/split:output:3*
T0*'
_output_shapes
:����������
!backward_lstm_93/lstm_cell/Relu_1Relu$backward_lstm_93/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
 backward_lstm_93/lstm_cell/mul_2Mul(backward_lstm_93/lstm_cell/Sigmoid_2:y:0/backward_lstm_93/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������
.backward_lstm_93/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
 backward_lstm_93/TensorArrayV2_1TensorListReserve7backward_lstm_93/TensorArrayV2_1/element_shape:output:0)backward_lstm_93/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���W
backward_lstm_93/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_93/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������e
#backward_lstm_93/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
backward_lstm_93/whileWhile,backward_lstm_93/while/loop_counter:output:02backward_lstm_93/while/maximum_iterations:output:0backward_lstm_93/time:output:0)backward_lstm_93/TensorArrayV2_1:handle:0backward_lstm_93/zeros:output:0!backward_lstm_93/zeros_1:output:0)backward_lstm_93/strided_slice_1:output:0Hbackward_lstm_93/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_93_lstm_cell_matmul_readvariableop_resource;backward_lstm_93_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_93_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$backward_lstm_93_while_body_55913036*0
cond(R&
$backward_lstm_93_while_cond_55913035*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
Abackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
3backward_lstm_93/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_93/while:output:3Jbackward_lstm_93/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0y
&backward_lstm_93/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������r
(backward_lstm_93/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_93/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
 backward_lstm_93/strided_slice_3StridedSlice<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_93/strided_slice_3/stack:output:01backward_lstm_93/strided_slice_3/stack_1:output:01backward_lstm_93/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskv
!backward_lstm_93/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
backward_lstm_93/transpose_1	Transpose<backward_lstm_93/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_93/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������l
backward_lstm_93/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:�
	ReverseV2	ReverseV2 backward_lstm_93/transpose_1:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :������������������M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2forward_lstm_93/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :������������������k
IdentityIdentityconcat:output:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp2^backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_93/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_93/while1^forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_93/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'���������������������������: : : : : : 2f
1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp0backward_lstm_93/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_93/whilebackward_lstm_93/while2d
0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_93/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp/forward_lstm_93/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_93/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_93/whileforward_lstm_93/while:g c
=
_output_shapes+
):'���������������������������
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
1__inference_sequential_408_layer_call_fn_55909729
dense_595_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_595_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909711s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:���������
)
_user_specified_namedense_595_input:($
"
_user_specified_name
55909723:($
"
_user_specified_name
55909725
�
q
G__inference_lambda_99_layer_call_and_return_conditional_losses_55911802

inputs
inputs_1
identity|
einsum/EinsumEinsuminputsinputs_1*
N*
T0*"
_output_shapes
:*
equationabc,abcd->abdY
IdentityIdentityeinsum/Einsum:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :::J F
"
_output_shapes
:
 
_user_specified_nameinputs:NJ
&
_output_shapes
:
 
_user_specified_nameinputs
�	
�
3__inference_backward_lstm_93_layer_call_fn_55914577

inputs
unknown: 
	unknown_0: 
	unknown_1: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55911278|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
55914569:($
"
_user_specified_name
55914571:($
"
_user_specified_name
55914573
�	
�
while_cond_55910398
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55910398___redundant_placeholder06
2while_while_cond_55910398___redundant_placeholder16
2while_while_cond_55910398___redundant_placeholder26
2while_while_cond_55910398___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :���������:���������: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
�

g
H__inference_dropout_93_layer_call_and_return_conditional_losses_55913722

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?_
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
:b
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*"
_output_shapes
:*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*"
_output_shapes
:\
IdentityIdentitydropout/SelectV2:output:0*
T0*"
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
�L
�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55911278

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identity�� lstm_cell/BiasAdd/ReadVariableOp�lstm_cell/MatMul/ReadVariableOp�!lstm_cell/MatMul_1/ReadVariableOp�whileI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'���������������������������R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::��_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: �
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'����������������������������
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:������������������*
shrink_axis_mask�
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0�
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:���������q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:���������b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:���������}
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:���������r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:���������_
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :���������:���������: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55911194*
condR
while_cond_55911193*K
output_shapes:
8: : : : :���������:���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'���������������������������: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�8
�
while_body_55914306
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0B
0while_lstm_cell_matmul_readvariableop_resource_0: D
2while_lstm_cell_matmul_1_readvariableop_resource_0: ?
1while_lstm_cell_biasadd_readvariableop_resource_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor@
.while_lstm_cell_matmul_readvariableop_resource: B
0while_lstm_cell_matmul_1_readvariableop_resource: =
/while_lstm_cell_biasadd_readvariableop_resource: ��&while/lstm_cell/BiasAdd/ReadVariableOp�%while/lstm_cell/MatMul/ReadVariableOp�'while/lstm_cell/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"���������
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:������������������*
element_dtype0�
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0�
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� �
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:��������� �
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0�
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:���������:���������:���������:���������*
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:����������
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:���������n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:����������
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:���������v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:���������k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:����������
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:���������v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :���������:���������: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
�
�
,__inference_dense_594_layer_call_fn_55912393

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_594_layer_call_and_return_conditional_losses_55911386j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs:($
"
_user_specified_name
55912387:($
"
_user_specified_name
55912389
�
�
G__inference_dense_605_layer_call_and_return_conditional_losses_55913877

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:���������e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:���������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
6
input_4+
serving_default_input_4:04
	dense_611'
StatefulPartitionedCall:0tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures
#_self_saveable_object_factories"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
#$_self_saveable_object_factories"
_tf_keras_layer
M
%	keras_api
#&_self_saveable_object_factories"
_tf_keras_layer
M
'	keras_api
#(_self_saveable_object_factories"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
#/_self_saveable_object_factories"
_tf_keras_layer
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
#6_self_saveable_object_factories"
_tf_keras_layer
M
7	keras_api
#8_self_saveable_object_factories"
_tf_keras_layer
�
9layer_with_weights-0
9layer-0
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
#@_self_saveable_object_factories"
_tf_keras_sequential
�
Alayer_with_weights-0
Alayer-0
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
#H_self_saveable_object_factories"
_tf_keras_sequential
�
Ilayer_with_weights-0
Ilayer-0
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
#P_self_saveable_object_factories"
_tf_keras_sequential
�
Qlayer_with_weights-0
Qlayer-0
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses
#X_self_saveable_object_factories"
_tf_keras_sequential
M
Y	keras_api
#Z_self_saveable_object_factories"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
#a_self_saveable_object_factories"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
hforward_layer
ibackward_layer
#j_self_saveable_object_factories"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
q_random_generator
#r_self_saveable_object_factories"
_tf_keras_layer
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
#y_self_saveable_object_factories"
_tf_keras_layer
�
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories"
_tf_keras_layer
�
"0
#1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
�
"0
#1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_topk_bilstm_moe_layer_call_fn_55912198
2__inference_topk_bilstm_moe_layer_call_fn_55912239�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55911782
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55912157�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
#__inference__wrapped_model_55909671input_4"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_594_layer_call_fn_55912393�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_594_layer_call_and_return_conditional_losses_55912408�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 2dense_594/kernel
:2dense_594/bias
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_lambda_99_layer_call_fn_55912414
,__inference_lambda_99_layer_call_fn_55912420�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_lambda_99_layer_call_and_return_conditional_losses_55912426
G__inference_lambda_99_layer_call_and_return_conditional_losses_55912432�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_lambda_100_layer_call_fn_55912438
-__inference_lambda_100_layer_call_fn_55912444�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_lambda_100_layer_call_and_return_conditional_losses_55912450
H__inference_lambda_100_layer_call_and_return_conditional_losses_55912456�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_sequential_408_layer_call_fn_55909729
1__inference_sequential_408_layer_call_fn_55909738�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909711
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909720�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_sequential_413_layer_call_fn_55909805
1__inference_sequential_413_layer_call_fn_55909814�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909787
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909796�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_sequential_418_layer_call_fn_55909881
1__inference_sequential_418_layer_call_fn_55909890�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909863
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909872�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias
$�_self_saveable_object_factories"
_tf_keras_layer
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
1__inference_sequential_423_layer_call_fn_55909957
1__inference_sequential_423_layer_call_fn_55909966�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909939
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909948�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_lambda_101_layer_call_fn_55912462
-__inference_lambda_101_layer_call_fn_55912468�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_lambda_101_layer_call_and_return_conditional_losses_55912474
H__inference_lambda_101_layer_call_and_return_conditional_losses_55912480�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
3__inference_bidirectional_93_layer_call_fn_55912497
3__inference_bidirectional_93_layer_call_fn_55912514
3__inference_bidirectional_93_layer_call_fn_55912531
3__inference_bidirectional_93_layer_call_fn_55912548�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55912836
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913124
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913412
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913700�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
	�cell
�
state_spec
$�_self_saveable_object_factories"
_tf_keras_rnn_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
	�cell
�
state_spec
$�_self_saveable_object_factories"
_tf_keras_rnn_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
-__inference_dropout_93_layer_call_fn_55913705
-__inference_dropout_93_layer_call_fn_55913710�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
H__inference_dropout_93_layer_call_and_return_conditional_losses_55913722
H__inference_dropout_93_layer_call_and_return_conditional_losses_55913727�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
D
$�_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_flatten_93_layer_call_fn_55913732�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_flatten_93_layer_call_and_return_conditional_losses_55913738�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_611_layer_call_fn_55913747�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_611_layer_call_and_return_conditional_losses_55913757�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!	�2dense_611/kernel
:2dense_611/bias
 "
trackable_dict_wrapper
": 2dense_595/kernel
:2dense_595/bias
": 2dense_600/kernel
:2dense_600/bias
": 2dense_605/kernel
:2dense_605/bias
": 2dense_610/kernel
:2dense_610/bias
C:A 21bidirectional_93/forward_lstm_93/lstm_cell/kernel
M:K 2;bidirectional_93/forward_lstm_93/lstm_cell/recurrent_kernel
=:; 2/bidirectional_93/forward_lstm_93/lstm_cell/bias
D:B 22bidirectional_93/backward_lstm_93/lstm_cell/kernel
N:L 2<bidirectional_93/backward_lstm_93/lstm_cell/recurrent_kernel
>:< 20bidirectional_93/backward_lstm_93/lstm_cell/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_topk_bilstm_moe_layer_call_fn_55912198input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_topk_bilstm_moe_layer_call_fn_55912239input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55911782input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55912157input_4"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_55912384input_4"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_594_layer_call_fn_55912393inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_594_layer_call_and_return_conditional_losses_55912408inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_lambda_99_layer_call_fn_55912414inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_lambda_99_layer_call_fn_55912420inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_lambda_99_layer_call_and_return_conditional_losses_55912426inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_lambda_99_layer_call_and_return_conditional_losses_55912432inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_lambda_100_layer_call_fn_55912438inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_lambda_100_layer_call_fn_55912444inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_lambda_100_layer_call_and_return_conditional_losses_55912450inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_lambda_100_layer_call_and_return_conditional_losses_55912456inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_595_layer_call_fn_55913766�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_595_layer_call_and_return_conditional_losses_55913797�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
90"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_sequential_408_layer_call_fn_55909729dense_595_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_408_layer_call_fn_55909738dense_595_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909711dense_595_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909720dense_595_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_600_layer_call_fn_55913806�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_600_layer_call_and_return_conditional_losses_55913837�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_sequential_413_layer_call_fn_55909805dense_600_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_413_layer_call_fn_55909814dense_600_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909787dense_600_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909796dense_600_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_605_layer_call_fn_55913846�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_605_layer_call_and_return_conditional_losses_55913877�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_sequential_418_layer_call_fn_55909881dense_605_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_418_layer_call_fn_55909890dense_605_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909863dense_605_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909872dense_605_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_dense_610_layer_call_fn_55913886�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_dense_610_layer_call_and_return_conditional_losses_55913917�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_sequential_423_layer_call_fn_55909957dense_610_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
1__inference_sequential_423_layer_call_fn_55909966dense_610_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909939dense_610_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909948dense_610_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_lambda_101_layer_call_fn_55912462inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_lambda_101_layer_call_fn_55912468inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_lambda_101_layer_call_and_return_conditional_losses_55912474inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_lambda_101_layer_call_and_return_conditional_losses_55912480inputs_0inputs_1"�
���
FullArgSpec)
args!�
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_bidirectional_93_layer_call_fn_55912497inputs_0"�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_bidirectional_93_layer_call_fn_55912514inputs_0"�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_bidirectional_93_layer_call_fn_55912531inputs"�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_bidirectional_93_layer_call_fn_55912548inputs"�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55912836inputs_0"�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913124inputs_0"�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913412inputs"�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913700inputs"�
���
FullArgSpecG
args?�<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaults�
p 

 

 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
8
�0
�1
�2"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
2__inference_forward_lstm_93_layer_call_fn_55913928
2__inference_forward_lstm_93_layer_call_fn_55913939
2__inference_forward_lstm_93_layer_call_fn_55913950
2__inference_forward_lstm_93_layer_call_fn_55913961�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914104
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914247
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914390
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914533�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
D
$�_self_saveable_object_factories"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�
state_size
�kernel
�recurrent_kernel
	�bias
$�_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
�0
�1
�2"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�states
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
3__inference_backward_lstm_93_layer_call_fn_55914544
3__inference_backward_lstm_93_layer_call_fn_55914555
3__inference_backward_lstm_93_layer_call_fn_55914566
3__inference_backward_lstm_93_layer_call_fn_55914577�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55914722
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55914867
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55915012
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55915157�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
D
$�_self_saveable_object_factories"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator
�
state_size
�kernel
�recurrent_kernel
	�bias
$�_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_dropout_93_layer_call_fn_55913705inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_dropout_93_layer_call_fn_55913710inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_93_layer_call_and_return_conditional_losses_55913722inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_dropout_93_layer_call_and_return_conditional_losses_55913727inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_flatten_93_layer_call_fn_55913732inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_flatten_93_layer_call_and_return_conditional_losses_55913738inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_611_layer_call_fn_55913747inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_611_layer_call_and_return_conditional_losses_55913757inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_595_layer_call_fn_55913766inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_595_layer_call_and_return_conditional_losses_55913797inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_600_layer_call_fn_55913806inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_600_layer_call_and_return_conditional_losses_55913837inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_605_layer_call_fn_55913846inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_605_layer_call_and_return_conditional_losses_55913877inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_610_layer_call_fn_55913886inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_610_layer_call_and_return_conditional_losses_55913917inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_forward_lstm_93_layer_call_fn_55913928inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_forward_lstm_93_layer_call_fn_55913939inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_forward_lstm_93_layer_call_fn_55913950inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
2__inference_forward_lstm_93_layer_call_fn_55913961inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914104inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914247inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914390inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914533inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_dict_wrapper
8
�0
�1
�2"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_lstm_cell_layer_call_fn_55915174
,__inference_lstm_cell_layer_call_fn_55915191�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915223
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915255�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
D
$�_self_saveable_object_factories"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_backward_lstm_93_layer_call_fn_55914544inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_backward_lstm_93_layer_call_fn_55914555inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_backward_lstm_93_layer_call_fn_55914566inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_backward_lstm_93_layer_call_fn_55914577inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55914722inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55914867inputs_0"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55915012inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55915157inputs"�
���
FullArgSpec:
args2�/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_dict_wrapper
8
�0
�1
�2"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_lstm_cell_layer_call_fn_55915272
,__inference_lstm_cell_layer_call_fn_55915289�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915321
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915353�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
D
$�_self_saveable_object_factories"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_lstm_cell_layer_call_fn_55915174inputsstates_0states_1"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_lstm_cell_layer_call_fn_55915191inputsstates_0states_1"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915223inputsstates_0states_1"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915255inputsstates_0states_1"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_lstm_cell_layer_call_fn_55915272inputsstates_0states_1"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_lstm_cell_layer_call_fn_55915289inputsstates_0states_1"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915321inputsstates_0states_1"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915353inputsstates_0states_1"�
���
FullArgSpec+
args#� 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_dict_wrapper�
#__inference__wrapped_model_55909671""#����������������+�(
!�
�
input_4
� ",�)
'
	dense_611�
	dense_611�
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55914722����O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� "9�6
/�,
tensor_0������������������
� �
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55914867����O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� "9�6
/�,
tensor_0������������������
� �
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55915012����Q�N
G�D
6�3
inputs'���������������������������

 
p

 
� "9�6
/�,
tensor_0������������������
� �
N__inference_backward_lstm_93_layer_call_and_return_conditional_losses_55915157����Q�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "9�6
/�,
tensor_0������������������
� �
3__inference_backward_lstm_93_layer_call_fn_55914544����O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� ".�+
unknown�������������������
3__inference_backward_lstm_93_layer_call_fn_55914555����O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� ".�+
unknown�������������������
3__inference_backward_lstm_93_layer_call_fn_55914566����Q�N
G�D
6�3
inputs'���������������������������

 
p

 
� ".�+
unknown�������������������
3__inference_backward_lstm_93_layer_call_fn_55914577����Q�N
G�D
6�3
inputs'���������������������������

 
p 

 
� ".�+
unknown�������������������
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55912836�������\�Y
R�O
=�:
8�5
inputs_0'���������������������������
p

 

 

 
� "9�6
/�,
tensor_0������������������
� �
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913124�������\�Y
R�O
=�:
8�5
inputs_0'���������������������������
p 

 

 

 
� "9�6
/�,
tensor_0������������������
� �
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913412s������:�7
0�-
�
inputs
p

 

 

 
� "'�$
�
tensor_0
� �
N__inference_bidirectional_93_layer_call_and_return_conditional_losses_55913700s������:�7
0�-
�
inputs
p 

 

 

 
� "'�$
�
tensor_0
� �
3__inference_bidirectional_93_layer_call_fn_55912497�������\�Y
R�O
=�:
8�5
inputs_0'���������������������������
p

 

 

 
� ".�+
unknown�������������������
3__inference_bidirectional_93_layer_call_fn_55912514�������\�Y
R�O
=�:
8�5
inputs_0'���������������������������
p 

 

 

 
� ".�+
unknown�������������������
3__inference_bidirectional_93_layer_call_fn_55912531h������:�7
0�-
�
inputs
p

 

 

 
� "�
unknown�
3__inference_bidirectional_93_layer_call_fn_55912548h������:�7
0�-
�
inputs
p 

 

 

 
� "�
unknown�
G__inference_dense_594_layer_call_and_return_conditional_losses_55912408Y"#*�'
 �
�
inputs
� "'�$
�
tensor_0
� ~
,__inference_dense_594_layer_call_fn_55912393N"#*�'
 �
�
inputs
� "�
unknown�
G__inference_dense_595_layer_call_and_return_conditional_losses_55913797m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
,__inference_dense_595_layer_call_fn_55913766b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
G__inference_dense_600_layer_call_and_return_conditional_losses_55913837m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
,__inference_dense_600_layer_call_fn_55913806b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
G__inference_dense_605_layer_call_and_return_conditional_losses_55913877m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
,__inference_dense_605_layer_call_fn_55913846b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
G__inference_dense_610_layer_call_and_return_conditional_losses_55913917m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
,__inference_dense_610_layer_call_fn_55913886b��3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
G__inference_dense_611_layer_call_and_return_conditional_losses_55913757T��'�$
�
�
inputs	�
� "#� 
�
tensor_0
� y
,__inference_dense_611_layer_call_fn_55913747I��'�$
�
�
inputs	�
� "�
unknown�
H__inference_dropout_93_layer_call_and_return_conditional_losses_55913722Y.�+
$�!
�
inputs
p
� "'�$
�
tensor_0
� �
H__inference_dropout_93_layer_call_and_return_conditional_losses_55913727Y.�+
$�!
�
inputs
p 
� "'�$
�
tensor_0
� 
-__inference_dropout_93_layer_call_fn_55913705N.�+
$�!
�
inputs
p
� "�
unknown
-__inference_dropout_93_layer_call_fn_55913710N.�+
$�!
�
inputs
p 
� "�
unknown�
H__inference_flatten_93_layer_call_and_return_conditional_losses_55913738R*�'
 �
�
inputs
� "$�!
�
tensor_0	�
� x
-__inference_flatten_93_layer_call_fn_55913732G*�'
 �
�
inputs
� "�
unknown	��
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914104����O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� "9�6
/�,
tensor_0������������������
� �
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914247����O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� "9�6
/�,
tensor_0������������������
� �
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914390����Q�N
G�D
6�3
inputs'���������������������������

 
p

 
� "9�6
/�,
tensor_0������������������
� �
M__inference_forward_lstm_93_layer_call_and_return_conditional_losses_55914533����Q�N
G�D
6�3
inputs'���������������������������

 
p 

 
� "9�6
/�,
tensor_0������������������
� �
2__inference_forward_lstm_93_layer_call_fn_55913928����O�L
E�B
4�1
/�,
inputs_0������������������

 
p

 
� ".�+
unknown�������������������
2__inference_forward_lstm_93_layer_call_fn_55913939����O�L
E�B
4�1
/�,
inputs_0������������������

 
p 

 
� ".�+
unknown�������������������
2__inference_forward_lstm_93_layer_call_fn_55913950����Q�N
G�D
6�3
inputs'���������������������������

 
p

 
� ".�+
unknown�������������������
2__inference_forward_lstm_93_layer_call_fn_55913961����Q�N
G�D
6�3
inputs'���������������������������

 
p 

 
� ".�+
unknown�������������������
H__inference_lambda_100_layer_call_and_return_conditional_losses_55912450�X�U
N�K
A�>
�
inputs_0
�
inputs_1

 
p
� "+�(
!�
tensor_0
� �
H__inference_lambda_100_layer_call_and_return_conditional_losses_55912456�X�U
N�K
A�>
�
inputs_0
�
inputs_1

 
p 
� "+�(
!�
tensor_0
� �
-__inference_lambda_100_layer_call_fn_55912438|X�U
N�K
A�>
�
inputs_0
�
inputs_1

 
p
� " �
unknown�
-__inference_lambda_100_layer_call_fn_55912444|X�U
N�K
A�>
�
inputs_0
�
inputs_1

 
p 
� " �
unknown�
H__inference_lambda_101_layer_call_and_return_conditional_losses_55912474�\�Y
R�O
E�B
!�
inputs_0
�
inputs_1

 
p
� "'�$
�
tensor_0
� �
H__inference_lambda_101_layer_call_and_return_conditional_losses_55912480�\�Y
R�O
E�B
!�
inputs_0
�
inputs_1

 
p 
� "'�$
�
tensor_0
� �
-__inference_lambda_101_layer_call_fn_55912462|\�Y
R�O
E�B
!�
inputs_0
�
inputs_1

 
p
� "�
unknown�
-__inference_lambda_101_layer_call_fn_55912468|\�Y
R�O
E�B
!�
inputs_0
�
inputs_1

 
p 
� "�
unknown�
G__inference_lambda_99_layer_call_and_return_conditional_losses_55912426�\�Y
R�O
E�B
�
inputs_0
!�
inputs_1

 
p
� "'�$
�
tensor_0
� �
G__inference_lambda_99_layer_call_and_return_conditional_losses_55912432�\�Y
R�O
E�B
�
inputs_0
!�
inputs_1

 
p 
� "'�$
�
tensor_0
� �
,__inference_lambda_99_layer_call_fn_55912414|\�Y
R�O
E�B
�
inputs_0
!�
inputs_1

 
p
� "�
unknown�
,__inference_lambda_99_layer_call_fn_55912420|\�Y
R�O
E�B
�
inputs_0
!�
inputs_1

 
p 
� "�
unknown�
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915223������}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915255������}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915321������}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55915353������}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "���
~�{
$�!

tensor_0_0���������
S�P
&�#
tensor_0_1_0���������
&�#
tensor_0_1_1���������
� �
,__inference_lstm_cell_layer_call_fn_55915174������}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
,__inference_lstm_cell_layer_call_fn_55915191������}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
,__inference_lstm_cell_layer_call_fn_55915272������}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
,__inference_lstm_cell_layer_call_fn_55915289������}
v�s
 �
inputs���������
K�H
"�
states_0���������
"�
states_1���������
p 
� "x�u
"�
tensor_0���������
O�L
$�!

tensor_1_0���������
$�!

tensor_1_1����������
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909711~��D�A
:�7
-�*
dense_595_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
L__inference_sequential_408_layer_call_and_return_conditional_losses_55909720~��D�A
:�7
-�*
dense_595_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
1__inference_sequential_408_layer_call_fn_55909729s��D�A
:�7
-�*
dense_595_input���������
p

 
� "%�"
unknown����������
1__inference_sequential_408_layer_call_fn_55909738s��D�A
:�7
-�*
dense_595_input���������
p 

 
� "%�"
unknown����������
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909787~��D�A
:�7
-�*
dense_600_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
L__inference_sequential_413_layer_call_and_return_conditional_losses_55909796~��D�A
:�7
-�*
dense_600_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
1__inference_sequential_413_layer_call_fn_55909805s��D�A
:�7
-�*
dense_600_input���������
p

 
� "%�"
unknown����������
1__inference_sequential_413_layer_call_fn_55909814s��D�A
:�7
-�*
dense_600_input���������
p 

 
� "%�"
unknown����������
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909863~��D�A
:�7
-�*
dense_605_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
L__inference_sequential_418_layer_call_and_return_conditional_losses_55909872~��D�A
:�7
-�*
dense_605_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
1__inference_sequential_418_layer_call_fn_55909881s��D�A
:�7
-�*
dense_605_input���������
p

 
� "%�"
unknown����������
1__inference_sequential_418_layer_call_fn_55909890s��D�A
:�7
-�*
dense_605_input���������
p 

 
� "%�"
unknown����������
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909939~��D�A
:�7
-�*
dense_610_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
L__inference_sequential_423_layer_call_and_return_conditional_losses_55909948~��D�A
:�7
-�*
dense_610_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
1__inference_sequential_423_layer_call_fn_55909957s��D�A
:�7
-�*
dense_610_input���������
p

 
� "%�"
unknown����������
1__inference_sequential_423_layer_call_fn_55909966s��D�A
:�7
-�*
dense_610_input���������
p 

 
� "%�"
unknown����������
&__inference_signature_wrapper_55912384�""#����������������6�3
� 
,�)
'
input_4�
input_4",�)
'
	dense_611�
	dense_611�
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55911782~""#����������������3�0
)�&
�
input_4
p

 
� "#� 
�
tensor_0
� �
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55912157~""#����������������3�0
)�&
�
input_4
p 

 
� "#� 
�
tensor_0
� �
2__inference_topk_bilstm_moe_layer_call_fn_55912198s""#����������������3�0
)�&
�
input_4
p

 
� "�
unknown�
2__inference_topk_bilstm_moe_layer_call_fn_55912239s""#����������������3�0
)�&
�
input_4
p 

 
� "�
unknown