��
��
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
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
 �"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
�
Adam/v/dense_3068/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3068/bias
}
*Adam/v/dense_3068/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3068/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3068/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3068/bias
}
*Adam/m/dense_3068/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3068/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3068/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/v/dense_3068/kernel
�
,Adam/v/dense_3068/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3068/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_3068/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/m/dense_3068/kernel
�
,Adam/m/dense_3068/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3068/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense_3067/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3067/bias
}
*Adam/v/dense_3067/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3067/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3067/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3067/bias
}
*Adam/m/dense_3067/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3067/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3067/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_3067/kernel
�
,Adam/v/dense_3067/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3067/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_3067/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_3067/kernel
�
,Adam/m/dense_3067/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3067/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_3066/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3066/bias
}
*Adam/v/dense_3066/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3066/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3066/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3066/bias
}
*Adam/m/dense_3066/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3066/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3066/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_3066/kernel
�
,Adam/v/dense_3066/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3066/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_3066/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_3066/kernel
�
,Adam/m/dense_3066/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3066/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_3065/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3065/bias
}
*Adam/v/dense_3065/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3065/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3065/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3065/bias
}
*Adam/m/dense_3065/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3065/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3065/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_3065/kernel
�
,Adam/v/dense_3065/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3065/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_3065/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_3065/kernel
�
,Adam/m/dense_3065/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3065/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_3059/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3059/bias
}
*Adam/v/dense_3059/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3059/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3059/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3059/bias
}
*Adam/m/dense_3059/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3059/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3059/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_3059/kernel
�
,Adam/v/dense_3059/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3059/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_3059/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_3059/kernel
�
,Adam/m/dense_3059/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3059/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_3053/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3053/bias
}
*Adam/v/dense_3053/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3053/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3053/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3053/bias
}
*Adam/m/dense_3053/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3053/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3053/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_3053/kernel
�
,Adam/v/dense_3053/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3053/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_3053/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_3053/kernel
�
,Adam/m/dense_3053/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3053/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_3047/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3047/bias
}
*Adam/v/dense_3047/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3047/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3047/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3047/bias
}
*Adam/m/dense_3047/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3047/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3047/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_3047/kernel
�
,Adam/v/dense_3047/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3047/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_3047/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_3047/kernel
�
,Adam/m/dense_3047/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3047/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_3041/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3041/bias
}
*Adam/v/dense_3041/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3041/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3041/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3041/bias
}
*Adam/m/dense_3041/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3041/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3041/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_3041/kernel
�
,Adam/v/dense_3041/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3041/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_3041/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_3041/kernel
�
,Adam/m/dense_3041/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3041/kernel*
_output_shapes

:*
dtype0
�
Adam/v/dense_3040/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/dense_3040/bias
}
*Adam/v/dense_3040/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3040/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_3040/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/dense_3040/bias
}
*Adam/m/dense_3040/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3040/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_3040/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/v/dense_3040/kernel
�
,Adam/v/dense_3040/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3040/kernel*
_output_shapes

:*
dtype0
�
Adam/m/dense_3040/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameAdam/m/dense_3040/kernel
�
,Adam/m/dense_3040/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3040/kernel*
_output_shapes

:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
v
dense_3065/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3065/bias
o
#dense_3065/bias/Read/ReadVariableOpReadVariableOpdense_3065/bias*
_output_shapes
:*
dtype0
~
dense_3065/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3065/kernel
w
%dense_3065/kernel/Read/ReadVariableOpReadVariableOpdense_3065/kernel*
_output_shapes

:*
dtype0
v
dense_3059/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3059/bias
o
#dense_3059/bias/Read/ReadVariableOpReadVariableOpdense_3059/bias*
_output_shapes
:*
dtype0
~
dense_3059/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3059/kernel
w
%dense_3059/kernel/Read/ReadVariableOpReadVariableOpdense_3059/kernel*
_output_shapes

:*
dtype0
v
dense_3053/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3053/bias
o
#dense_3053/bias/Read/ReadVariableOpReadVariableOpdense_3053/bias*
_output_shapes
:*
dtype0
~
dense_3053/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3053/kernel
w
%dense_3053/kernel/Read/ReadVariableOpReadVariableOpdense_3053/kernel*
_output_shapes

:*
dtype0
v
dense_3047/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3047/bias
o
#dense_3047/bias/Read/ReadVariableOpReadVariableOpdense_3047/bias*
_output_shapes
:*
dtype0
~
dense_3047/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3047/kernel
w
%dense_3047/kernel/Read/ReadVariableOpReadVariableOpdense_3047/kernel*
_output_shapes

:*
dtype0
v
dense_3041/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3041/bias
o
#dense_3041/bias/Read/ReadVariableOpReadVariableOpdense_3041/bias*
_output_shapes
:*
dtype0
~
dense_3041/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3041/kernel
w
%dense_3041/kernel/Read/ReadVariableOpReadVariableOpdense_3041/kernel*
_output_shapes

:*
dtype0
v
dense_3068/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3068/bias
o
#dense_3068/bias/Read/ReadVariableOpReadVariableOpdense_3068/bias*
_output_shapes
:*
dtype0

dense_3068/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*"
shared_namedense_3068/kernel
x
%dense_3068/kernel/Read/ReadVariableOpReadVariableOpdense_3068/kernel*
_output_shapes
:	�*
dtype0
v
dense_3067/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3067/bias
o
#dense_3067/bias/Read/ReadVariableOpReadVariableOpdense_3067/bias*
_output_shapes
:*
dtype0
~
dense_3067/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3067/kernel
w
%dense_3067/kernel/Read/ReadVariableOpReadVariableOpdense_3067/kernel*
_output_shapes

:*
dtype0
v
dense_3066/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3066/bias
o
#dense_3066/bias/Read/ReadVariableOpReadVariableOpdense_3066/bias*
_output_shapes
:*
dtype0
~
dense_3066/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3066/kernel
w
%dense_3066/kernel/Read/ReadVariableOpReadVariableOpdense_3066/kernel*
_output_shapes

:*
dtype0
v
dense_3040/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_3040/bias
o
#dense_3040/bias/Read/ReadVariableOpReadVariableOpdense_3040/bias*
_output_shapes
:*
dtype0
~
dense_3040/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*"
shared_namedense_3040/kernel
w
%dense_3040/kernel/Read/ReadVariableOpReadVariableOpdense_3040/kernel*
_output_shapes

:*
dtype0
t
serving_default_input_layerPlaceholder*"
_output_shapes
:*
dtype0*
shape:
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerdense_3040/kerneldense_3040/biasdense_3041/kerneldense_3041/biasdense_3047/kerneldense_3047/biasdense_3053/kerneldense_3053/biasdense_3059/kerneldense_3059/biasdense_3065/kerneldense_3065/biasdense_3066/kerneldense_3066/biasdense_3067/kerneldense_3067/biasdense_3068/kerneldense_3068/bias*
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
&__inference_signature_wrapper_14165950

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� B�
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
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*

%	keras_api* 

&	keras_api* 

'	keras_api* 

(	keras_api* 

)	keras_api* 
�
*layer_with_weights-0
*layer-0
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
�
1layer_with_weights-0
1layer-0
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses*
�
8layer_with_weights-0
8layer-0
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
�
?layer_with_weights-0
?layer-0
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses*
�
Flayer_with_weights-0
Flayer-0
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*

M	keras_api* 

N	keras_api* 
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias*
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias*
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e_random_generator* 
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses* 
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias*
�
#0
$1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11
U12
V13
]14
^15
r16
s17*
�
#0
$1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11
U12
V13
]14
^15
r16
s17*
* 
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

#0
$1*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEdense_3040/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_3040/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
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

tkernel
ubias*

t0
u1*

t0
u1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

vkernel
wbias*

v0
w1*

v0
w1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

xkernel
ybias*

x0
y1*

x0
y1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

zkernel
{bias*

z0
{1*

z0
{1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

|kernel
}bias*

|0
}1*

|0
}1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 

U0
V1*

U0
V1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEdense_3066/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_3066/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

]0
^1*

]0
^1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEdense_3067/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_3067/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

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
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

r0
s1*

r0
s1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEdense_3068/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_3068/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEdense_3041/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3041/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEdense_3047/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3047/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEdense_3053/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3053/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEdense_3059/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3059/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEdense_3065/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_3065/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
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
16
17
18*
$
�0
�1
�2
�3*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17*
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
t0
u1*

t0
u1*
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

*0*
* 
* 
* 
* 
* 
* 
* 

v0
w1*

v0
w1*
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

10*
* 
* 
* 
* 
* 
* 
* 

x0
y1*

x0
y1*
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

80*
* 
* 
* 
* 
* 
* 
* 

z0
{1*

z0
{1*
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

?0*
* 
* 
* 
* 
* 
* 
* 

|0
}1*

|0
}1*
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

F0*
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
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
c]
VARIABLE_VALUEAdam/m/dense_3040/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_3040/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3040/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3040/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_3041/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/dense_3041/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3041/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3041/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/dense_3047/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_3047/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_3047/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_3047/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_3053/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_3053/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_3053/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_3053/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_3059/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_3059/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_3059/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_3059/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_3065/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_3065/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_3065/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_3065/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_3066/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_3066/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_3066/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_3066/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_3067/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_3067/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_3067/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_3067/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/dense_3068/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/dense_3068/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_3068/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_3068/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
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

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_3040/kerneldense_3040/biasdense_3066/kerneldense_3066/biasdense_3067/kerneldense_3067/biasdense_3068/kerneldense_3068/biasdense_3041/kerneldense_3041/biasdense_3047/kerneldense_3047/biasdense_3053/kerneldense_3053/biasdense_3059/kerneldense_3059/biasdense_3065/kerneldense_3065/bias	iterationlearning_rateAdam/m/dense_3040/kernelAdam/v/dense_3040/kernelAdam/m/dense_3040/biasAdam/v/dense_3040/biasAdam/m/dense_3041/kernelAdam/v/dense_3041/kernelAdam/m/dense_3041/biasAdam/v/dense_3041/biasAdam/m/dense_3047/kernelAdam/v/dense_3047/kernelAdam/m/dense_3047/biasAdam/v/dense_3047/biasAdam/m/dense_3053/kernelAdam/v/dense_3053/kernelAdam/m/dense_3053/biasAdam/v/dense_3053/biasAdam/m/dense_3059/kernelAdam/v/dense_3059/kernelAdam/m/dense_3059/biasAdam/v/dense_3059/biasAdam/m/dense_3065/kernelAdam/v/dense_3065/kernelAdam/m/dense_3065/biasAdam/v/dense_3065/biasAdam/m/dense_3066/kernelAdam/v/dense_3066/kernelAdam/m/dense_3066/biasAdam/v/dense_3066/biasAdam/m/dense_3067/kernelAdam/v/dense_3067/kernelAdam/m/dense_3067/biasAdam/v/dense_3067/biasAdam/m/dense_3068/kernelAdam/v/dense_3068/kernelAdam/m/dense_3068/biasAdam/v/dense_3068/biastotal_3count_3total_2count_2total_1count_1totalcountConst*M
TinF
D2B*
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
!__inference__traced_save_14166684
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3040/kerneldense_3040/biasdense_3066/kerneldense_3066/biasdense_3067/kerneldense_3067/biasdense_3068/kerneldense_3068/biasdense_3041/kerneldense_3041/biasdense_3047/kerneldense_3047/biasdense_3053/kerneldense_3053/biasdense_3059/kerneldense_3059/biasdense_3065/kerneldense_3065/bias	iterationlearning_rateAdam/m/dense_3040/kernelAdam/v/dense_3040/kernelAdam/m/dense_3040/biasAdam/v/dense_3040/biasAdam/m/dense_3041/kernelAdam/v/dense_3041/kernelAdam/m/dense_3041/biasAdam/v/dense_3041/biasAdam/m/dense_3047/kernelAdam/v/dense_3047/kernelAdam/m/dense_3047/biasAdam/v/dense_3047/biasAdam/m/dense_3053/kernelAdam/v/dense_3053/kernelAdam/m/dense_3053/biasAdam/v/dense_3053/biasAdam/m/dense_3059/kernelAdam/v/dense_3059/kernelAdam/m/dense_3059/biasAdam/v/dense_3059/biasAdam/m/dense_3065/kernelAdam/v/dense_3065/kernelAdam/m/dense_3065/biasAdam/v/dense_3065/biasAdam/m/dense_3066/kernelAdam/v/dense_3066/kernelAdam/m/dense_3066/biasAdam/v/dense_3066/biasAdam/m/dense_3067/kernelAdam/v/dense_3067/kernelAdam/m/dense_3067/biasAdam/v/dense_3067/biasAdam/m/dense_3068/kernelAdam/v/dense_3068/kernelAdam/m/dense_3068/biasAdam/v/dense_3068/biastotal_3count_3total_2count_2total_1count_1totalcount*L
TinE
C2A*
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
$__inference__traced_restore_14166885��
�
�
2__inference_sequential_2372_layer_call_fn_14165387
dense_3053_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3053_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165369s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165383:($
"
_user_specified_name
14165381:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3053_input
�	
�
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165286
dense_3047_input%
dense_3047_14165280:!
dense_3047_14165282:
identity��"dense_3047/StatefulPartitionedCall�
"dense_3047/StatefulPartitionedCallStatefulPartitionedCalldense_3047_inputdense_3047_14165280dense_3047_14165282*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3047_layer_call_and_return_conditional_losses_14165265~
IdentityIdentity+dense_3047/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3047/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3047/StatefulPartitionedCall"dense_3047/StatefulPartitionedCall:($
"
_user_specified_name
14165282:($
"
_user_specified_name
14165280:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3047_input
�
�
-__inference_dense_3059_layer_call_fn_14166207

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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3059_layer_call_and_return_conditional_losses_14165431s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14166203:($
"
_user_specified_name
14166201:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

h
I__inference_dropout_170_layer_call_and_return_conditional_losses_14166043

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
�
�
&__inference_signature_wrapper_14165950
input_layer
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	�

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
#__inference__wrapped_model_14165147f
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165946:($
"
_user_specified_name
14165944:($
"
_user_specified_name
14165942:($
"
_user_specified_name
14165940:($
"
_user_specified_name
14165938:($
"
_user_specified_name
14165936:($
"
_user_specified_name
14165934:($
"
_user_specified_name
14165932:(
$
"
_user_specified_name
14165930:(	$
"
_user_specified_name
14165928:($
"
_user_specified_name
14165926:($
"
_user_specified_name
14165924:($
"
_user_specified_name
14165922:($
"
_user_specified_name
14165920:($
"
_user_specified_name
14165918:($
"
_user_specified_name
14165916:($
"
_user_specified_name
14165914:($
"
_user_specified_name
14165912:O K
"
_output_shapes
:
%
_user_specified_nameinput_layer
�
�
H__inference_dense_3067_layer_call_and_return_conditional_losses_14166021

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     p
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:K
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
:\
IdentityIdentityRelu:activations:0^NoOp*
T0*"
_output_shapes
:V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
-__inference_dense_3047_layer_call_fn_14166127

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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3047_layer_call_and_return_conditional_losses_14165265s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14166123:($
"
_user_specified_name
14166121:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
H__inference_dense_3068_layer_call_and_return_conditional_losses_14165694

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
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:G C

_output_shapes
:	�
 
_user_specified_nameinputs
��
�
#__inference__wrapped_model_14165147
input_layerG
5topk_moe_dense_3040_tensordot_readvariableop_resource:A
3topk_moe_dense_3040_biasadd_readvariableop_resource:W
Etopk_moe_sequential_2360_dense_3041_tensordot_readvariableop_resource:Q
Ctopk_moe_sequential_2360_dense_3041_biasadd_readvariableop_resource:W
Etopk_moe_sequential_2366_dense_3047_tensordot_readvariableop_resource:Q
Ctopk_moe_sequential_2366_dense_3047_biasadd_readvariableop_resource:W
Etopk_moe_sequential_2372_dense_3053_tensordot_readvariableop_resource:Q
Ctopk_moe_sequential_2372_dense_3053_biasadd_readvariableop_resource:W
Etopk_moe_sequential_2378_dense_3059_tensordot_readvariableop_resource:Q
Ctopk_moe_sequential_2378_dense_3059_biasadd_readvariableop_resource:W
Etopk_moe_sequential_2384_dense_3065_tensordot_readvariableop_resource:Q
Ctopk_moe_sequential_2384_dense_3065_biasadd_readvariableop_resource:G
5topk_moe_dense_3066_tensordot_readvariableop_resource:A
3topk_moe_dense_3066_biasadd_readvariableop_resource:G
5topk_moe_dense_3067_tensordot_readvariableop_resource:A
3topk_moe_dense_3067_biasadd_readvariableop_resource:E
2topk_moe_dense_3068_matmul_readvariableop_resource:	�A
3topk_moe_dense_3068_biasadd_readvariableop_resource:
identity��*topk_moe/dense_3040/BiasAdd/ReadVariableOp�,topk_moe/dense_3040/Tensordot/ReadVariableOp�*topk_moe/dense_3066/BiasAdd/ReadVariableOp�,topk_moe/dense_3066/Tensordot/ReadVariableOp�*topk_moe/dense_3067/BiasAdd/ReadVariableOp�,topk_moe/dense_3067/Tensordot/ReadVariableOp�*topk_moe/dense_3068/BiasAdd/ReadVariableOp�)topk_moe/dense_3068/MatMul/ReadVariableOp�:topk_moe/sequential_2360/dense_3041/BiasAdd/ReadVariableOp�<topk_moe/sequential_2360/dense_3041/Tensordot/ReadVariableOp�:topk_moe/sequential_2366/dense_3047/BiasAdd/ReadVariableOp�<topk_moe/sequential_2366/dense_3047/Tensordot/ReadVariableOp�:topk_moe/sequential_2372/dense_3053/BiasAdd/ReadVariableOp�<topk_moe/sequential_2372/dense_3053/Tensordot/ReadVariableOp�:topk_moe/sequential_2378/dense_3059/BiasAdd/ReadVariableOp�<topk_moe/sequential_2378/dense_3059/Tensordot/ReadVariableOp�:topk_moe/sequential_2384/dense_3065/BiasAdd/ReadVariableOp�<topk_moe/sequential_2384/dense_3065/Tensordot/ReadVariableOp�
,topk_moe/dense_3040/Tensordot/ReadVariableOpReadVariableOp5topk_moe_dense_3040_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0|
+topk_moe/dense_3040/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
%topk_moe/dense_3040/Tensordot/ReshapeReshapeinput_layer4topk_moe/dense_3040/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
$topk_moe/dense_3040/Tensordot/MatMulMatMul.topk_moe/dense_3040/Tensordot/Reshape:output:04topk_moe/dense_3040/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	�x
#topk_moe/dense_3040/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
topk_moe/dense_3040/TensordotReshape.topk_moe/dense_3040/Tensordot/MatMul:product:0,topk_moe/dense_3040/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
*topk_moe/dense_3040/BiasAdd/ReadVariableOpReadVariableOp3topk_moe_dense_3040_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
topk_moe/dense_3040/BiasAddBiasAdd&topk_moe/dense_3040/Tensordot:output:02topk_moe/dense_3040/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:y
topk_moe/dense_3040/SoftmaxSoftmax$topk_moe/dense_3040/BiasAdd:output:0*
T0*"
_output_shapes
:d
"topk_moe/tf.math.top_k_80/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :�
 topk_moe/tf.math.top_k_80/TopKV2TopKV2%topk_moe/dense_3040/Softmax:softmax:0+topk_moe/tf.math.top_k_80/TopKV2/k:output:0*
T0*0
_output_shapes
::l
'topk_moe/tf.one_hot_80/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?m
(topk_moe/tf.one_hot_80/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    f
$topk_moe/tf.one_hot_80/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
topk_moe/tf.one_hot_80/one_hotOneHot*topk_moe/tf.math.top_k_80/TopKV2:indices:0-topk_moe/tf.one_hot_80/one_hot/depth:output:00topk_moe/tf.one_hot_80/one_hot/on_value:output:01topk_moe/tf.one_hot_80/one_hot/off_value:output:0*
TI0*
T0*&
_output_shapes
:�
$topk_moe/tf.einsum_330/einsum/EinsumEinsum)topk_moe/tf.math.top_k_80/TopKV2:values:0'topk_moe/tf.one_hot_80/one_hot:output:0*
N*
T0*"
_output_shapes
:*
equationabc,abcd->abd�
$topk_moe/tf.einsum_331/einsum/EinsumEinsuminput_layer-topk_moe/tf.einsum_330/einsum/Einsum:output:0*
N*
T0*&
_output_shapes
:*
equationabc,abd->dabc�
topk_moe/tf.unstack_80/unstackUnpack-topk_moe/tf.einsum_331/einsum/Einsum:output:0*
T0*Z
_output_shapesH
F:::::*	
num�
<topk_moe/sequential_2360/dense_3041/Tensordot/ReadVariableOpReadVariableOpEtopk_moe_sequential_2360_dense_3041_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
;topk_moe/sequential_2360/dense_3041/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
5topk_moe/sequential_2360/dense_3041/Tensordot/ReshapeReshape'topk_moe/tf.unstack_80/unstack:output:0Dtopk_moe/sequential_2360/dense_3041/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
4topk_moe/sequential_2360/dense_3041/Tensordot/MatMulMatMul>topk_moe/sequential_2360/dense_3041/Tensordot/Reshape:output:0Dtopk_moe/sequential_2360/dense_3041/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
3topk_moe/sequential_2360/dense_3041/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
-topk_moe/sequential_2360/dense_3041/TensordotReshape>topk_moe/sequential_2360/dense_3041/Tensordot/MatMul:product:0<topk_moe/sequential_2360/dense_3041/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
:topk_moe/sequential_2360/dense_3041/BiasAdd/ReadVariableOpReadVariableOpCtopk_moe_sequential_2360_dense_3041_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+topk_moe/sequential_2360/dense_3041/BiasAddBiasAdd6topk_moe/sequential_2360/dense_3041/Tensordot:output:0Btopk_moe/sequential_2360/dense_3041/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
(topk_moe/sequential_2360/dense_3041/ReluRelu4topk_moe/sequential_2360/dense_3041/BiasAdd:output:0*
T0*"
_output_shapes
:�
<topk_moe/sequential_2366/dense_3047/Tensordot/ReadVariableOpReadVariableOpEtopk_moe_sequential_2366_dense_3047_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
;topk_moe/sequential_2366/dense_3047/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
5topk_moe/sequential_2366/dense_3047/Tensordot/ReshapeReshape'topk_moe/tf.unstack_80/unstack:output:1Dtopk_moe/sequential_2366/dense_3047/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
4topk_moe/sequential_2366/dense_3047/Tensordot/MatMulMatMul>topk_moe/sequential_2366/dense_3047/Tensordot/Reshape:output:0Dtopk_moe/sequential_2366/dense_3047/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
3topk_moe/sequential_2366/dense_3047/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
-topk_moe/sequential_2366/dense_3047/TensordotReshape>topk_moe/sequential_2366/dense_3047/Tensordot/MatMul:product:0<topk_moe/sequential_2366/dense_3047/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
:topk_moe/sequential_2366/dense_3047/BiasAdd/ReadVariableOpReadVariableOpCtopk_moe_sequential_2366_dense_3047_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+topk_moe/sequential_2366/dense_3047/BiasAddBiasAdd6topk_moe/sequential_2366/dense_3047/Tensordot:output:0Btopk_moe/sequential_2366/dense_3047/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
(topk_moe/sequential_2366/dense_3047/ReluRelu4topk_moe/sequential_2366/dense_3047/BiasAdd:output:0*
T0*"
_output_shapes
:�
<topk_moe/sequential_2372/dense_3053/Tensordot/ReadVariableOpReadVariableOpEtopk_moe_sequential_2372_dense_3053_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
;topk_moe/sequential_2372/dense_3053/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
5topk_moe/sequential_2372/dense_3053/Tensordot/ReshapeReshape'topk_moe/tf.unstack_80/unstack:output:2Dtopk_moe/sequential_2372/dense_3053/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
4topk_moe/sequential_2372/dense_3053/Tensordot/MatMulMatMul>topk_moe/sequential_2372/dense_3053/Tensordot/Reshape:output:0Dtopk_moe/sequential_2372/dense_3053/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
3topk_moe/sequential_2372/dense_3053/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
-topk_moe/sequential_2372/dense_3053/TensordotReshape>topk_moe/sequential_2372/dense_3053/Tensordot/MatMul:product:0<topk_moe/sequential_2372/dense_3053/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
:topk_moe/sequential_2372/dense_3053/BiasAdd/ReadVariableOpReadVariableOpCtopk_moe_sequential_2372_dense_3053_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+topk_moe/sequential_2372/dense_3053/BiasAddBiasAdd6topk_moe/sequential_2372/dense_3053/Tensordot:output:0Btopk_moe/sequential_2372/dense_3053/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
(topk_moe/sequential_2372/dense_3053/ReluRelu4topk_moe/sequential_2372/dense_3053/BiasAdd:output:0*
T0*"
_output_shapes
:�
<topk_moe/sequential_2378/dense_3059/Tensordot/ReadVariableOpReadVariableOpEtopk_moe_sequential_2378_dense_3059_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
;topk_moe/sequential_2378/dense_3059/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
5topk_moe/sequential_2378/dense_3059/Tensordot/ReshapeReshape'topk_moe/tf.unstack_80/unstack:output:3Dtopk_moe/sequential_2378/dense_3059/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
4topk_moe/sequential_2378/dense_3059/Tensordot/MatMulMatMul>topk_moe/sequential_2378/dense_3059/Tensordot/Reshape:output:0Dtopk_moe/sequential_2378/dense_3059/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
3topk_moe/sequential_2378/dense_3059/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
-topk_moe/sequential_2378/dense_3059/TensordotReshape>topk_moe/sequential_2378/dense_3059/Tensordot/MatMul:product:0<topk_moe/sequential_2378/dense_3059/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
:topk_moe/sequential_2378/dense_3059/BiasAdd/ReadVariableOpReadVariableOpCtopk_moe_sequential_2378_dense_3059_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+topk_moe/sequential_2378/dense_3059/BiasAddBiasAdd6topk_moe/sequential_2378/dense_3059/Tensordot:output:0Btopk_moe/sequential_2378/dense_3059/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
(topk_moe/sequential_2378/dense_3059/ReluRelu4topk_moe/sequential_2378/dense_3059/BiasAdd:output:0*
T0*"
_output_shapes
:�
<topk_moe/sequential_2384/dense_3065/Tensordot/ReadVariableOpReadVariableOpEtopk_moe_sequential_2384_dense_3065_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0�
;topk_moe/sequential_2384/dense_3065/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
5topk_moe/sequential_2384/dense_3065/Tensordot/ReshapeReshape'topk_moe/tf.unstack_80/unstack:output:4Dtopk_moe/sequential_2384/dense_3065/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
4topk_moe/sequential_2384/dense_3065/Tensordot/MatMulMatMul>topk_moe/sequential_2384/dense_3065/Tensordot/Reshape:output:0Dtopk_moe/sequential_2384/dense_3065/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
3topk_moe/sequential_2384/dense_3065/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
-topk_moe/sequential_2384/dense_3065/TensordotReshape>topk_moe/sequential_2384/dense_3065/Tensordot/MatMul:product:0<topk_moe/sequential_2384/dense_3065/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
:topk_moe/sequential_2384/dense_3065/BiasAdd/ReadVariableOpReadVariableOpCtopk_moe_sequential_2384_dense_3065_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
+topk_moe/sequential_2384/dense_3065/BiasAddBiasAdd6topk_moe/sequential_2384/dense_3065/Tensordot:output:0Btopk_moe/sequential_2384/dense_3065/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
(topk_moe/sequential_2384/dense_3065/ReluRelu4topk_moe/sequential_2384/dense_3065/BiasAdd:output:0*
T0*"
_output_shapes
:�
topk_moe/tf.stack_170/stackPack6topk_moe/sequential_2360/dense_3041/Relu:activations:06topk_moe/sequential_2366/dense_3047/Relu:activations:06topk_moe/sequential_2372/dense_3053/Relu:activations:06topk_moe/sequential_2378/dense_3059/Relu:activations:06topk_moe/sequential_2384/dense_3065/Relu:activations:0*
N*
T0*&
_output_shapes
:*

axis�
$topk_moe/tf.einsum_332/einsum/EinsumEinsum$topk_moe/tf.stack_170/stack:output:0-topk_moe/tf.einsum_330/einsum/Einsum:output:0*
N*
T0*"
_output_shapes
:*
equationabcd,ace->acd�
,topk_moe/dense_3066/Tensordot/ReadVariableOpReadVariableOp5topk_moe_dense_3066_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0|
+topk_moe/dense_3066/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
%topk_moe/dense_3066/Tensordot/ReshapeReshape-topk_moe/tf.einsum_332/einsum/Einsum:output:04topk_moe/dense_3066/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
$topk_moe/dense_3066/Tensordot/MatMulMatMul.topk_moe/dense_3066/Tensordot/Reshape:output:04topk_moe/dense_3066/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	�x
#topk_moe/dense_3066/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
topk_moe/dense_3066/TensordotReshape.topk_moe/dense_3066/Tensordot/MatMul:product:0,topk_moe/dense_3066/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
*topk_moe/dense_3066/BiasAdd/ReadVariableOpReadVariableOp3topk_moe_dense_3066_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
topk_moe/dense_3066/BiasAddBiasAdd&topk_moe/dense_3066/Tensordot:output:02topk_moe/dense_3066/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:�
,topk_moe/dense_3067/Tensordot/ReadVariableOpReadVariableOp5topk_moe_dense_3067_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0|
+topk_moe/dense_3067/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     �
%topk_moe/dense_3067/Tensordot/ReshapeReshape$topk_moe/dense_3066/BiasAdd:output:04topk_moe/dense_3067/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
$topk_moe/dense_3067/Tensordot/MatMulMatMul.topk_moe/dense_3067/Tensordot/Reshape:output:04topk_moe/dense_3067/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	�x
#topk_moe/dense_3067/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         �
topk_moe/dense_3067/TensordotReshape.topk_moe/dense_3067/Tensordot/MatMul:product:0,topk_moe/dense_3067/Tensordot/shape:output:0*
T0*"
_output_shapes
:�
*topk_moe/dense_3067/BiasAdd/ReadVariableOpReadVariableOp3topk_moe_dense_3067_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
topk_moe/dense_3067/BiasAddBiasAdd&topk_moe/dense_3067/Tensordot:output:02topk_moe/dense_3067/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:s
topk_moe/dense_3067/ReluRelu$topk_moe/dense_3067/BiasAdd:output:0*
T0*"
_output_shapes
:~
topk_moe/dropout_170/IdentityIdentity&topk_moe/dense_3067/Relu:activations:0*
T0*"
_output_shapes
:k
topk_moe/flatten_170/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  �
topk_moe/flatten_170/ReshapeReshape&topk_moe/dropout_170/Identity:output:0#topk_moe/flatten_170/Const:output:0*
T0*
_output_shapes
:	��
)topk_moe/dense_3068/MatMul/ReadVariableOpReadVariableOp2topk_moe_dense_3068_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
topk_moe/dense_3068/MatMulMatMul%topk_moe/flatten_170/Reshape:output:01topk_moe/dense_3068/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
*topk_moe/dense_3068/BiasAdd/ReadVariableOpReadVariableOp3topk_moe_dense_3068_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
topk_moe/dense_3068/BiasAddBiasAdd$topk_moe/dense_3068/MatMul:product:02topk_moe/dense_3068/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:j
IdentityIdentity$topk_moe/dense_3068/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp+^topk_moe/dense_3040/BiasAdd/ReadVariableOp-^topk_moe/dense_3040/Tensordot/ReadVariableOp+^topk_moe/dense_3066/BiasAdd/ReadVariableOp-^topk_moe/dense_3066/Tensordot/ReadVariableOp+^topk_moe/dense_3067/BiasAdd/ReadVariableOp-^topk_moe/dense_3067/Tensordot/ReadVariableOp+^topk_moe/dense_3068/BiasAdd/ReadVariableOp*^topk_moe/dense_3068/MatMul/ReadVariableOp;^topk_moe/sequential_2360/dense_3041/BiasAdd/ReadVariableOp=^topk_moe/sequential_2360/dense_3041/Tensordot/ReadVariableOp;^topk_moe/sequential_2366/dense_3047/BiasAdd/ReadVariableOp=^topk_moe/sequential_2366/dense_3047/Tensordot/ReadVariableOp;^topk_moe/sequential_2372/dense_3053/BiasAdd/ReadVariableOp=^topk_moe/sequential_2372/dense_3053/Tensordot/ReadVariableOp;^topk_moe/sequential_2378/dense_3059/BiasAdd/ReadVariableOp=^topk_moe/sequential_2378/dense_3059/Tensordot/ReadVariableOp;^topk_moe/sequential_2384/dense_3065/BiasAdd/ReadVariableOp=^topk_moe/sequential_2384/dense_3065/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 2X
*topk_moe/dense_3040/BiasAdd/ReadVariableOp*topk_moe/dense_3040/BiasAdd/ReadVariableOp2\
,topk_moe/dense_3040/Tensordot/ReadVariableOp,topk_moe/dense_3040/Tensordot/ReadVariableOp2X
*topk_moe/dense_3066/BiasAdd/ReadVariableOp*topk_moe/dense_3066/BiasAdd/ReadVariableOp2\
,topk_moe/dense_3066/Tensordot/ReadVariableOp,topk_moe/dense_3066/Tensordot/ReadVariableOp2X
*topk_moe/dense_3067/BiasAdd/ReadVariableOp*topk_moe/dense_3067/BiasAdd/ReadVariableOp2\
,topk_moe/dense_3067/Tensordot/ReadVariableOp,topk_moe/dense_3067/Tensordot/ReadVariableOp2X
*topk_moe/dense_3068/BiasAdd/ReadVariableOp*topk_moe/dense_3068/BiasAdd/ReadVariableOp2V
)topk_moe/dense_3068/MatMul/ReadVariableOp)topk_moe/dense_3068/MatMul/ReadVariableOp2x
:topk_moe/sequential_2360/dense_3041/BiasAdd/ReadVariableOp:topk_moe/sequential_2360/dense_3041/BiasAdd/ReadVariableOp2|
<topk_moe/sequential_2360/dense_3041/Tensordot/ReadVariableOp<topk_moe/sequential_2360/dense_3041/Tensordot/ReadVariableOp2x
:topk_moe/sequential_2366/dense_3047/BiasAdd/ReadVariableOp:topk_moe/sequential_2366/dense_3047/BiasAdd/ReadVariableOp2|
<topk_moe/sequential_2366/dense_3047/Tensordot/ReadVariableOp<topk_moe/sequential_2366/dense_3047/Tensordot/ReadVariableOp2x
:topk_moe/sequential_2372/dense_3053/BiasAdd/ReadVariableOp:topk_moe/sequential_2372/dense_3053/BiasAdd/ReadVariableOp2|
<topk_moe/sequential_2372/dense_3053/Tensordot/ReadVariableOp<topk_moe/sequential_2372/dense_3053/Tensordot/ReadVariableOp2x
:topk_moe/sequential_2378/dense_3059/BiasAdd/ReadVariableOp:topk_moe/sequential_2378/dense_3059/BiasAdd/ReadVariableOp2|
<topk_moe/sequential_2378/dense_3059/Tensordot/ReadVariableOp<topk_moe/sequential_2378/dense_3059/Tensordot/ReadVariableOp2x
:topk_moe/sequential_2384/dense_3065/BiasAdd/ReadVariableOp:topk_moe/sequential_2384/dense_3065/BiasAdd/ReadVariableOp2|
<topk_moe/sequential_2384/dense_3065/Tensordot/ReadVariableOp<topk_moe/sequential_2384/dense_3065/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
"
_output_shapes
:
%
_user_specified_nameinput_layer
�	
�
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165535
dense_3065_input%
dense_3065_14165529:!
dense_3065_14165531:
identity��"dense_3065/StatefulPartitionedCall�
"dense_3065/StatefulPartitionedCallStatefulPartitionedCalldense_3065_inputdense_3065_14165529dense_3065_14165531*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3065_layer_call_and_return_conditional_losses_14165514~
IdentityIdentity+dense_3065/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3065/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3065/StatefulPartitionedCall"dense_3065/StatefulPartitionedCall:($
"
_user_specified_name
14165531:($
"
_user_specified_name
14165529:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3065_input
��
�9
!__inference__traced_save_14166684
file_prefix:
(read_disablecopyonread_dense_3040_kernel:6
(read_1_disablecopyonread_dense_3040_bias:<
*read_2_disablecopyonread_dense_3066_kernel:6
(read_3_disablecopyonread_dense_3066_bias:<
*read_4_disablecopyonread_dense_3067_kernel:6
(read_5_disablecopyonread_dense_3067_bias:=
*read_6_disablecopyonread_dense_3068_kernel:	�6
(read_7_disablecopyonread_dense_3068_bias:<
*read_8_disablecopyonread_dense_3041_kernel:6
(read_9_disablecopyonread_dense_3041_bias:=
+read_10_disablecopyonread_dense_3047_kernel:7
)read_11_disablecopyonread_dense_3047_bias:=
+read_12_disablecopyonread_dense_3053_kernel:7
)read_13_disablecopyonread_dense_3053_bias:=
+read_14_disablecopyonread_dense_3059_kernel:7
)read_15_disablecopyonread_dense_3059_bias:=
+read_16_disablecopyonread_dense_3065_kernel:7
)read_17_disablecopyonread_dense_3065_bias:-
#read_18_disablecopyonread_iteration:	 1
'read_19_disablecopyonread_learning_rate: D
2read_20_disablecopyonread_adam_m_dense_3040_kernel:D
2read_21_disablecopyonread_adam_v_dense_3040_kernel:>
0read_22_disablecopyonread_adam_m_dense_3040_bias:>
0read_23_disablecopyonread_adam_v_dense_3040_bias:D
2read_24_disablecopyonread_adam_m_dense_3041_kernel:D
2read_25_disablecopyonread_adam_v_dense_3041_kernel:>
0read_26_disablecopyonread_adam_m_dense_3041_bias:>
0read_27_disablecopyonread_adam_v_dense_3041_bias:D
2read_28_disablecopyonread_adam_m_dense_3047_kernel:D
2read_29_disablecopyonread_adam_v_dense_3047_kernel:>
0read_30_disablecopyonread_adam_m_dense_3047_bias:>
0read_31_disablecopyonread_adam_v_dense_3047_bias:D
2read_32_disablecopyonread_adam_m_dense_3053_kernel:D
2read_33_disablecopyonread_adam_v_dense_3053_kernel:>
0read_34_disablecopyonread_adam_m_dense_3053_bias:>
0read_35_disablecopyonread_adam_v_dense_3053_bias:D
2read_36_disablecopyonread_adam_m_dense_3059_kernel:D
2read_37_disablecopyonread_adam_v_dense_3059_kernel:>
0read_38_disablecopyonread_adam_m_dense_3059_bias:>
0read_39_disablecopyonread_adam_v_dense_3059_bias:D
2read_40_disablecopyonread_adam_m_dense_3065_kernel:D
2read_41_disablecopyonread_adam_v_dense_3065_kernel:>
0read_42_disablecopyonread_adam_m_dense_3065_bias:>
0read_43_disablecopyonread_adam_v_dense_3065_bias:D
2read_44_disablecopyonread_adam_m_dense_3066_kernel:D
2read_45_disablecopyonread_adam_v_dense_3066_kernel:>
0read_46_disablecopyonread_adam_m_dense_3066_bias:>
0read_47_disablecopyonread_adam_v_dense_3066_bias:D
2read_48_disablecopyonread_adam_m_dense_3067_kernel:D
2read_49_disablecopyonread_adam_v_dense_3067_kernel:>
0read_50_disablecopyonread_adam_m_dense_3067_bias:>
0read_51_disablecopyonread_adam_v_dense_3067_bias:E
2read_52_disablecopyonread_adam_m_dense_3068_kernel:	�E
2read_53_disablecopyonread_adam_v_dense_3068_kernel:	�>
0read_54_disablecopyonread_adam_m_dense_3068_bias:>
0read_55_disablecopyonread_adam_v_dense_3068_bias:+
!read_56_disablecopyonread_total_3: +
!read_57_disablecopyonread_count_3: +
!read_58_disablecopyonread_total_2: +
!read_59_disablecopyonread_count_2: +
!read_60_disablecopyonread_total_1: +
!read_61_disablecopyonread_count_1: )
read_62_disablecopyonread_total: )
read_63_disablecopyonread_count: 
savev2_const
identity_129��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: z
Read/DisableCopyOnReadDisableCopyOnRead(read_disablecopyonread_dense_3040_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp(read_disablecopyonread_dense_3040_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:|
Read_1/DisableCopyOnReadDisableCopyOnRead(read_1_disablecopyonread_dense_3040_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp(read_1_disablecopyonread_dense_3040_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_2/DisableCopyOnReadDisableCopyOnRead*read_2_disablecopyonread_dense_3066_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp*read_2_disablecopyonread_dense_3066_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:|
Read_3/DisableCopyOnReadDisableCopyOnRead(read_3_disablecopyonread_dense_3066_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp(read_3_disablecopyonread_dense_3066_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_4/DisableCopyOnReadDisableCopyOnRead*read_4_disablecopyonread_dense_3067_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp*read_4_disablecopyonread_dense_3067_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:|
Read_5/DisableCopyOnReadDisableCopyOnRead(read_5_disablecopyonread_dense_3067_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp(read_5_disablecopyonread_dense_3067_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_6/DisableCopyOnReadDisableCopyOnRead*read_6_disablecopyonread_dense_3068_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp*read_6_disablecopyonread_dense_3068_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	�|
Read_7/DisableCopyOnReadDisableCopyOnRead(read_7_disablecopyonread_dense_3068_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp(read_7_disablecopyonread_dense_3068_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_8/DisableCopyOnReadDisableCopyOnRead*read_8_disablecopyonread_dense_3041_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp*read_8_disablecopyonread_dense_3041_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
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

:|
Read_9/DisableCopyOnReadDisableCopyOnRead(read_9_disablecopyonread_dense_3041_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp(read_9_disablecopyonread_dense_3041_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
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
:�
Read_10/DisableCopyOnReadDisableCopyOnRead+read_10_disablecopyonread_dense_3047_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp+read_10_disablecopyonread_dense_3047_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
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

:~
Read_11/DisableCopyOnReadDisableCopyOnRead)read_11_disablecopyonread_dense_3047_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp)read_11_disablecopyonread_dense_3047_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
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
Read_12/DisableCopyOnReadDisableCopyOnRead+read_12_disablecopyonread_dense_3053_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp+read_12_disablecopyonread_dense_3053_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:~
Read_13/DisableCopyOnReadDisableCopyOnRead)read_13_disablecopyonread_dense_3053_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp)read_13_disablecopyonread_dense_3053_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnRead+read_14_disablecopyonread_dense_3059_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp+read_14_disablecopyonread_dense_3059_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:~
Read_15/DisableCopyOnReadDisableCopyOnRead)read_15_disablecopyonread_dense_3059_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp)read_15_disablecopyonread_dense_3059_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead+read_16_disablecopyonread_dense_3065_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp+read_16_disablecopyonread_dense_3065_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:~
Read_17/DisableCopyOnReadDisableCopyOnRead)read_17_disablecopyonread_dense_3065_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp)read_17_disablecopyonread_dense_3065_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_18/DisableCopyOnReadDisableCopyOnRead#read_18_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp#read_18_disablecopyonread_iteration^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_learning_rate^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_20/DisableCopyOnReadDisableCopyOnRead2read_20_disablecopyonread_adam_m_dense_3040_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp2read_20_disablecopyonread_adam_m_dense_3040_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_21/DisableCopyOnReadDisableCopyOnRead2read_21_disablecopyonread_adam_v_dense_3040_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp2read_21_disablecopyonread_adam_v_dense_3040_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_22/DisableCopyOnReadDisableCopyOnRead0read_22_disablecopyonread_adam_m_dense_3040_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp0read_22_disablecopyonread_adam_m_dense_3040_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_23/DisableCopyOnReadDisableCopyOnRead0read_23_disablecopyonread_adam_v_dense_3040_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp0read_23_disablecopyonread_adam_v_dense_3040_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_24/DisableCopyOnReadDisableCopyOnRead2read_24_disablecopyonread_adam_m_dense_3041_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp2read_24_disablecopyonread_adam_m_dense_3041_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_25/DisableCopyOnReadDisableCopyOnRead2read_25_disablecopyonread_adam_v_dense_3041_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp2read_25_disablecopyonread_adam_v_dense_3041_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_m_dense_3041_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_m_dense_3041_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_27/DisableCopyOnReadDisableCopyOnRead0read_27_disablecopyonread_adam_v_dense_3041_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp0read_27_disablecopyonread_adam_v_dense_3041_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_28/DisableCopyOnReadDisableCopyOnRead2read_28_disablecopyonread_adam_m_dense_3047_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp2read_28_disablecopyonread_adam_m_dense_3047_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_29/DisableCopyOnReadDisableCopyOnRead2read_29_disablecopyonread_adam_v_dense_3047_kernel"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp2read_29_disablecopyonread_adam_v_dense_3047_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_30/DisableCopyOnReadDisableCopyOnRead0read_30_disablecopyonread_adam_m_dense_3047_bias"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp0read_30_disablecopyonread_adam_m_dense_3047_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_31/DisableCopyOnReadDisableCopyOnRead0read_31_disablecopyonread_adam_v_dense_3047_bias"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp0read_31_disablecopyonread_adam_v_dense_3047_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_32/DisableCopyOnReadDisableCopyOnRead2read_32_disablecopyonread_adam_m_dense_3053_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp2read_32_disablecopyonread_adam_m_dense_3053_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_33/DisableCopyOnReadDisableCopyOnRead2read_33_disablecopyonread_adam_v_dense_3053_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp2read_33_disablecopyonread_adam_v_dense_3053_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_34/DisableCopyOnReadDisableCopyOnRead0read_34_disablecopyonread_adam_m_dense_3053_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp0read_34_disablecopyonread_adam_m_dense_3053_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_35/DisableCopyOnReadDisableCopyOnRead0read_35_disablecopyonread_adam_v_dense_3053_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp0read_35_disablecopyonread_adam_v_dense_3053_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_36/DisableCopyOnReadDisableCopyOnRead2read_36_disablecopyonread_adam_m_dense_3059_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp2read_36_disablecopyonread_adam_m_dense_3059_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_37/DisableCopyOnReadDisableCopyOnRead2read_37_disablecopyonread_adam_v_dense_3059_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp2read_37_disablecopyonread_adam_v_dense_3059_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_38/DisableCopyOnReadDisableCopyOnRead0read_38_disablecopyonread_adam_m_dense_3059_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp0read_38_disablecopyonread_adam_m_dense_3059_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_39/DisableCopyOnReadDisableCopyOnRead0read_39_disablecopyonread_adam_v_dense_3059_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp0read_39_disablecopyonread_adam_v_dense_3059_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_40/DisableCopyOnReadDisableCopyOnRead2read_40_disablecopyonread_adam_m_dense_3065_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp2read_40_disablecopyonread_adam_m_dense_3065_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_41/DisableCopyOnReadDisableCopyOnRead2read_41_disablecopyonread_adam_v_dense_3065_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp2read_41_disablecopyonread_adam_v_dense_3065_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_42/DisableCopyOnReadDisableCopyOnRead0read_42_disablecopyonread_adam_m_dense_3065_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp0read_42_disablecopyonread_adam_m_dense_3065_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnRead0read_43_disablecopyonread_adam_v_dense_3065_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp0read_43_disablecopyonread_adam_v_dense_3065_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnRead2read_44_disablecopyonread_adam_m_dense_3066_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp2read_44_disablecopyonread_adam_m_dense_3066_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_45/DisableCopyOnReadDisableCopyOnRead2read_45_disablecopyonread_adam_v_dense_3066_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp2read_45_disablecopyonread_adam_v_dense_3066_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_46/DisableCopyOnReadDisableCopyOnRead0read_46_disablecopyonread_adam_m_dense_3066_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp0read_46_disablecopyonread_adam_m_dense_3066_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_47/DisableCopyOnReadDisableCopyOnRead0read_47_disablecopyonread_adam_v_dense_3066_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp0read_47_disablecopyonread_adam_v_dense_3066_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_48/DisableCopyOnReadDisableCopyOnRead2read_48_disablecopyonread_adam_m_dense_3067_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp2read_48_disablecopyonread_adam_m_dense_3067_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_49/DisableCopyOnReadDisableCopyOnRead2read_49_disablecopyonread_adam_v_dense_3067_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp2read_49_disablecopyonread_adam_v_dense_3067_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_50/DisableCopyOnReadDisableCopyOnRead0read_50_disablecopyonread_adam_m_dense_3067_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp0read_50_disablecopyonread_adam_m_dense_3067_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_51/DisableCopyOnReadDisableCopyOnRead0read_51_disablecopyonread_adam_v_dense_3067_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp0read_51_disablecopyonread_adam_v_dense_3067_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_52/DisableCopyOnReadDisableCopyOnRead2read_52_disablecopyonread_adam_m_dense_3068_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp2read_52_disablecopyonread_adam_m_dense_3068_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_53/DisableCopyOnReadDisableCopyOnRead2read_53_disablecopyonread_adam_v_dense_3068_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp2read_53_disablecopyonread_adam_v_dense_3068_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_54/DisableCopyOnReadDisableCopyOnRead0read_54_disablecopyonread_adam_m_dense_3068_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp0read_54_disablecopyonread_adam_m_dense_3068_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_55/DisableCopyOnReadDisableCopyOnRead0read_55_disablecopyonread_adam_v_dense_3068_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp0read_55_disablecopyonread_adam_v_dense_3068_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_56/DisableCopyOnReadDisableCopyOnRead!read_56_disablecopyonread_total_3"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp!read_56_disablecopyonread_total_3^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_57/DisableCopyOnReadDisableCopyOnRead!read_57_disablecopyonread_count_3"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp!read_57_disablecopyonread_count_3^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_58/DisableCopyOnReadDisableCopyOnRead!read_58_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp!read_58_disablecopyonread_total_2^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_59/DisableCopyOnReadDisableCopyOnRead!read_59_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp!read_59_disablecopyonread_count_2^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_60/DisableCopyOnReadDisableCopyOnRead!read_60_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp!read_60_disablecopyonread_total_1^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_61/DisableCopyOnReadDisableCopyOnRead!read_61_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp!read_61_disablecopyonread_count_1^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_62/DisableCopyOnReadDisableCopyOnReadread_62_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOpread_62_disablecopyonread_total^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_63/DisableCopyOnReadDisableCopyOnReadread_63_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOpread_63_disablecopyonread_count^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *O
dtypesE
C2A	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_128Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_129IdentityIdentity_128:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_129Identity_129:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=A9

_output_shapes
: 

_user_specified_nameConst:%@!

_user_specified_namecount:%?!

_user_specified_nametotal:'>#
!
_user_specified_name	count_1:'=#
!
_user_specified_name	total_1:'<#
!
_user_specified_name	count_2:';#
!
_user_specified_name	total_2:':#
!
_user_specified_name	count_3:'9#
!
_user_specified_name	total_3:682
0
_user_specified_nameAdam/v/dense_3068/bias:672
0
_user_specified_nameAdam/m/dense_3068/bias:864
2
_user_specified_nameAdam/v/dense_3068/kernel:854
2
_user_specified_nameAdam/m/dense_3068/kernel:642
0
_user_specified_nameAdam/v/dense_3067/bias:632
0
_user_specified_nameAdam/m/dense_3067/bias:824
2
_user_specified_nameAdam/v/dense_3067/kernel:814
2
_user_specified_nameAdam/m/dense_3067/kernel:602
0
_user_specified_nameAdam/v/dense_3066/bias:6/2
0
_user_specified_nameAdam/m/dense_3066/bias:8.4
2
_user_specified_nameAdam/v/dense_3066/kernel:8-4
2
_user_specified_nameAdam/m/dense_3066/kernel:6,2
0
_user_specified_nameAdam/v/dense_3065/bias:6+2
0
_user_specified_nameAdam/m/dense_3065/bias:8*4
2
_user_specified_nameAdam/v/dense_3065/kernel:8)4
2
_user_specified_nameAdam/m/dense_3065/kernel:6(2
0
_user_specified_nameAdam/v/dense_3059/bias:6'2
0
_user_specified_nameAdam/m/dense_3059/bias:8&4
2
_user_specified_nameAdam/v/dense_3059/kernel:8%4
2
_user_specified_nameAdam/m/dense_3059/kernel:6$2
0
_user_specified_nameAdam/v/dense_3053/bias:6#2
0
_user_specified_nameAdam/m/dense_3053/bias:8"4
2
_user_specified_nameAdam/v/dense_3053/kernel:8!4
2
_user_specified_nameAdam/m/dense_3053/kernel:6 2
0
_user_specified_nameAdam/v/dense_3047/bias:62
0
_user_specified_nameAdam/m/dense_3047/bias:84
2
_user_specified_nameAdam/v/dense_3047/kernel:84
2
_user_specified_nameAdam/m/dense_3047/kernel:62
0
_user_specified_nameAdam/v/dense_3041/bias:62
0
_user_specified_nameAdam/m/dense_3041/bias:84
2
_user_specified_nameAdam/v/dense_3041/kernel:84
2
_user_specified_nameAdam/m/dense_3041/kernel:62
0
_user_specified_nameAdam/v/dense_3040/bias:62
0
_user_specified_nameAdam/m/dense_3040/bias:84
2
_user_specified_nameAdam/v/dense_3040/kernel:84
2
_user_specified_nameAdam/m/dense_3040/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:/+
)
_user_specified_namedense_3065/bias:1-
+
_user_specified_namedense_3065/kernel:/+
)
_user_specified_namedense_3059/bias:1-
+
_user_specified_namedense_3059/kernel:/+
)
_user_specified_namedense_3053/bias:1-
+
_user_specified_namedense_3053/kernel:/+
)
_user_specified_namedense_3047/bias:1-
+
_user_specified_namedense_3047/kernel:/
+
)
_user_specified_namedense_3041/bias:1	-
+
_user_specified_namedense_3041/kernel:/+
)
_user_specified_namedense_3068/bias:1-
+
_user_specified_namedense_3068/kernel:/+
)
_user_specified_namedense_3067/bias:1-
+
_user_specified_namedense_3067/kernel:/+
)
_user_specified_namedense_3066/bias:1-
+
_user_specified_namedense_3066/kernel:/+
)
_user_specified_namedense_3040/bias:1-
+
_user_specified_namedense_3040/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
H__inference_dense_3053_layer_call_and_return_conditional_losses_14166198

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_dropout_170_layer_call_fn_14166031

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
GPU 2J 8� *R
fMRK
I__inference_dropout_170_layer_call_and_return_conditional_losses_14165764[
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
�
�
-__inference_dense_3041_layer_call_fn_14166087

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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3041_layer_call_and_return_conditional_losses_14165182s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14166083:($
"
_user_specified_name
14166081:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�L
�	
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165773
input_layer%
dense_3040_14165704:!
dense_3040_14165706:*
sequential_2360_14165723:&
sequential_2360_14165725:*
sequential_2366_14165728:&
sequential_2366_14165730:*
sequential_2372_14165733:&
sequential_2372_14165735:*
sequential_2378_14165738:&
sequential_2378_14165740:*
sequential_2384_14165743:&
sequential_2384_14165745:%
dense_3066_14165750:!
dense_3066_14165752:%
dense_3067_14165755:!
dense_3067_14165757:&
dense_3068_14165767:	�!
dense_3068_14165769:
identity��"dense_3040/StatefulPartitionedCall�"dense_3066/StatefulPartitionedCall�"dense_3067/StatefulPartitionedCall�"dense_3068/StatefulPartitionedCall�'sequential_2360/StatefulPartitionedCall�'sequential_2366/StatefulPartitionedCall�'sequential_2372/StatefulPartitionedCall�'sequential_2378/StatefulPartitionedCall�'sequential_2384/StatefulPartitionedCall�
"dense_3040/StatefulPartitionedCallStatefulPartitionedCallinput_layerdense_3040_14165704dense_3040_14165706*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_3040_layer_call_and_return_conditional_losses_14165579[
tf.math.top_k_80/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.top_k_80/TopKV2TopKV2+dense_3040/StatefulPartitionedCall:output:0"tf.math.top_k_80/TopKV2/k:output:0*
T0*0
_output_shapes
::c
tf.one_hot_80/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
tf.one_hot_80/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ]
tf.one_hot_80/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
tf.one_hot_80/one_hotOneHot!tf.math.top_k_80/TopKV2:indices:0$tf.one_hot_80/one_hot/depth:output:0'tf.one_hot_80/one_hot/on_value:output:0(tf.one_hot_80/one_hot/off_value:output:0*
TI0*
T0*&
_output_shapes
:�
tf.einsum_330/einsum/EinsumEinsum tf.math.top_k_80/TopKV2:values:0tf.one_hot_80/one_hot:output:0*
N*
T0*"
_output_shapes
:*
equationabc,abcd->abd�
tf.einsum_331/einsum/EinsumEinsuminput_layer$tf.einsum_330/einsum/Einsum:output:0*
N*
T0*&
_output_shapes
:*
equationabc,abd->dabc�
tf.unstack_80/unstackUnpack$tf.einsum_331/einsum/Einsum:output:0*
T0*Z
_output_shapesH
F:::::*	
num�
'sequential_2360/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:0sequential_2360_14165723sequential_2360_14165725*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165203�
'sequential_2366/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:1sequential_2366_14165728sequential_2366_14165730*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165286�
'sequential_2372/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:2sequential_2372_14165733sequential_2372_14165735*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165369�
'sequential_2378/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:3sequential_2378_14165738sequential_2378_14165740*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165452�
'sequential_2384/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:4sequential_2384_14165743sequential_2384_14165745*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165535�
tf.stack_170/stackPack0sequential_2360/StatefulPartitionedCall:output:00sequential_2366/StatefulPartitionedCall:output:00sequential_2372/StatefulPartitionedCall:output:00sequential_2378/StatefulPartitionedCall:output:00sequential_2384/StatefulPartitionedCall:output:0*
N*
T0*&
_output_shapes
:*

axis�
tf.einsum_332/einsum/EinsumEinsumtf.stack_170/stack:output:0$tf.einsum_330/einsum/Einsum:output:0*
N*
T0*"
_output_shapes
:*
equationabcd,ace->acd�
"dense_3066/StatefulPartitionedCallStatefulPartitionedCall$tf.einsum_332/einsum/Einsum:output:0dense_3066_14165750dense_3066_14165752*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_3066_layer_call_and_return_conditional_losses_14165639�
"dense_3067/StatefulPartitionedCallStatefulPartitionedCall+dense_3066/StatefulPartitionedCall:output:0dense_3067_14165755dense_3067_14165757*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_3067_layer_call_and_return_conditional_losses_14165659�
dropout_170/PartitionedCallPartitionedCall+dense_3067/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *R
fMRK
I__inference_dropout_170_layer_call_and_return_conditional_losses_14165764�
flatten_170/PartitionedCallPartitionedCall$dropout_170/PartitionedCall:output:0*
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
GPU 2J 8� *R
fMRK
I__inference_flatten_170_layer_call_and_return_conditional_losses_14165683�
"dense_3068/StatefulPartitionedCallStatefulPartitionedCall$flatten_170/PartitionedCall:output:0dense_3068_14165767dense_3068_14165769*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3068_layer_call_and_return_conditional_losses_14165694q
IdentityIdentity+dense_3068/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp#^dense_3040/StatefulPartitionedCall#^dense_3066/StatefulPartitionedCall#^dense_3067/StatefulPartitionedCall#^dense_3068/StatefulPartitionedCall(^sequential_2360/StatefulPartitionedCall(^sequential_2366/StatefulPartitionedCall(^sequential_2372/StatefulPartitionedCall(^sequential_2378/StatefulPartitionedCall(^sequential_2384/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 2H
"dense_3040/StatefulPartitionedCall"dense_3040/StatefulPartitionedCall2H
"dense_3066/StatefulPartitionedCall"dense_3066/StatefulPartitionedCall2H
"dense_3067/StatefulPartitionedCall"dense_3067/StatefulPartitionedCall2H
"dense_3068/StatefulPartitionedCall"dense_3068/StatefulPartitionedCall2R
'sequential_2360/StatefulPartitionedCall'sequential_2360/StatefulPartitionedCall2R
'sequential_2366/StatefulPartitionedCall'sequential_2366/StatefulPartitionedCall2R
'sequential_2372/StatefulPartitionedCall'sequential_2372/StatefulPartitionedCall2R
'sequential_2378/StatefulPartitionedCall'sequential_2378/StatefulPartitionedCall2R
'sequential_2384/StatefulPartitionedCall'sequential_2384/StatefulPartitionedCall:($
"
_user_specified_name
14165769:($
"
_user_specified_name
14165767:($
"
_user_specified_name
14165757:($
"
_user_specified_name
14165755:($
"
_user_specified_name
14165752:($
"
_user_specified_name
14165750:($
"
_user_specified_name
14165745:($
"
_user_specified_name
14165743:(
$
"
_user_specified_name
14165740:(	$
"
_user_specified_name
14165738:($
"
_user_specified_name
14165735:($
"
_user_specified_name
14165733:($
"
_user_specified_name
14165730:($
"
_user_specified_name
14165728:($
"
_user_specified_name
14165725:($
"
_user_specified_name
14165723:($
"
_user_specified_name
14165706:($
"
_user_specified_name
14165704:O K
"
_output_shapes
:
%
_user_specified_nameinput_layer
�
�
H__inference_dense_3041_layer_call_and_return_conditional_losses_14166118

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_dense_3066_layer_call_fn_14165983

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_3066_layer_call_and_return_conditional_losses_14165639j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165979:($
"
_user_specified_name
14165977:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�M
�	
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165701
input_layer%
dense_3040_14165580:!
dense_3040_14165582:*
sequential_2360_14165599:&
sequential_2360_14165601:*
sequential_2366_14165604:&
sequential_2366_14165606:*
sequential_2372_14165609:&
sequential_2372_14165611:*
sequential_2378_14165614:&
sequential_2378_14165616:*
sequential_2384_14165619:&
sequential_2384_14165621:%
dense_3066_14165640:!
dense_3066_14165642:%
dense_3067_14165660:!
dense_3067_14165662:&
dense_3068_14165695:	�!
dense_3068_14165697:
identity��"dense_3040/StatefulPartitionedCall�"dense_3066/StatefulPartitionedCall�"dense_3067/StatefulPartitionedCall�"dense_3068/StatefulPartitionedCall�#dropout_170/StatefulPartitionedCall�'sequential_2360/StatefulPartitionedCall�'sequential_2366/StatefulPartitionedCall�'sequential_2372/StatefulPartitionedCall�'sequential_2378/StatefulPartitionedCall�'sequential_2384/StatefulPartitionedCall�
"dense_3040/StatefulPartitionedCallStatefulPartitionedCallinput_layerdense_3040_14165580dense_3040_14165582*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_3040_layer_call_and_return_conditional_losses_14165579[
tf.math.top_k_80/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :�
tf.math.top_k_80/TopKV2TopKV2+dense_3040/StatefulPartitionedCall:output:0"tf.math.top_k_80/TopKV2/k:output:0*
T0*0
_output_shapes
::c
tf.one_hot_80/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
tf.one_hot_80/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ]
tf.one_hot_80/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
tf.one_hot_80/one_hotOneHot!tf.math.top_k_80/TopKV2:indices:0$tf.one_hot_80/one_hot/depth:output:0'tf.one_hot_80/one_hot/on_value:output:0(tf.one_hot_80/one_hot/off_value:output:0*
TI0*
T0*&
_output_shapes
:�
tf.einsum_330/einsum/EinsumEinsum tf.math.top_k_80/TopKV2:values:0tf.one_hot_80/one_hot:output:0*
N*
T0*"
_output_shapes
:*
equationabc,abcd->abd�
tf.einsum_331/einsum/EinsumEinsuminput_layer$tf.einsum_330/einsum/Einsum:output:0*
N*
T0*&
_output_shapes
:*
equationabc,abd->dabc�
tf.unstack_80/unstackUnpack$tf.einsum_331/einsum/Einsum:output:0*
T0*Z
_output_shapesH
F:::::*	
num�
'sequential_2360/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:0sequential_2360_14165599sequential_2360_14165601*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165194�
'sequential_2366/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:1sequential_2366_14165604sequential_2366_14165606*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165277�
'sequential_2372/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:2sequential_2372_14165609sequential_2372_14165611*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165360�
'sequential_2378/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:3sequential_2378_14165614sequential_2378_14165616*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165443�
'sequential_2384/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_80/unstack:output:4sequential_2384_14165619sequential_2384_14165621*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165526�
tf.stack_170/stackPack0sequential_2360/StatefulPartitionedCall:output:00sequential_2366/StatefulPartitionedCall:output:00sequential_2372/StatefulPartitionedCall:output:00sequential_2378/StatefulPartitionedCall:output:00sequential_2384/StatefulPartitionedCall:output:0*
N*
T0*&
_output_shapes
:*

axis�
tf.einsum_332/einsum/EinsumEinsumtf.stack_170/stack:output:0$tf.einsum_330/einsum/Einsum:output:0*
N*
T0*"
_output_shapes
:*
equationabcd,ace->acd�
"dense_3066/StatefulPartitionedCallStatefulPartitionedCall$tf.einsum_332/einsum/Einsum:output:0dense_3066_14165640dense_3066_14165642*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_3066_layer_call_and_return_conditional_losses_14165639�
"dense_3067/StatefulPartitionedCallStatefulPartitionedCall+dense_3066/StatefulPartitionedCall:output:0dense_3067_14165660dense_3067_14165662*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_3067_layer_call_and_return_conditional_losses_14165659�
#dropout_170/StatefulPartitionedCallStatefulPartitionedCall+dense_3067/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *R
fMRK
I__inference_dropout_170_layer_call_and_return_conditional_losses_14165676�
flatten_170/PartitionedCallPartitionedCall,dropout_170/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *R
fMRK
I__inference_flatten_170_layer_call_and_return_conditional_losses_14165683�
"dense_3068/StatefulPartitionedCallStatefulPartitionedCall$flatten_170/PartitionedCall:output:0dense_3068_14165695dense_3068_14165697*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3068_layer_call_and_return_conditional_losses_14165694q
IdentityIdentity+dense_3068/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp#^dense_3040/StatefulPartitionedCall#^dense_3066/StatefulPartitionedCall#^dense_3067/StatefulPartitionedCall#^dense_3068/StatefulPartitionedCall$^dropout_170/StatefulPartitionedCall(^sequential_2360/StatefulPartitionedCall(^sequential_2366/StatefulPartitionedCall(^sequential_2372/StatefulPartitionedCall(^sequential_2378/StatefulPartitionedCall(^sequential_2384/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 2H
"dense_3040/StatefulPartitionedCall"dense_3040/StatefulPartitionedCall2H
"dense_3066/StatefulPartitionedCall"dense_3066/StatefulPartitionedCall2H
"dense_3067/StatefulPartitionedCall"dense_3067/StatefulPartitionedCall2H
"dense_3068/StatefulPartitionedCall"dense_3068/StatefulPartitionedCall2J
#dropout_170/StatefulPartitionedCall#dropout_170/StatefulPartitionedCall2R
'sequential_2360/StatefulPartitionedCall'sequential_2360/StatefulPartitionedCall2R
'sequential_2366/StatefulPartitionedCall'sequential_2366/StatefulPartitionedCall2R
'sequential_2372/StatefulPartitionedCall'sequential_2372/StatefulPartitionedCall2R
'sequential_2378/StatefulPartitionedCall'sequential_2378/StatefulPartitionedCall2R
'sequential_2384/StatefulPartitionedCall'sequential_2384/StatefulPartitionedCall:($
"
_user_specified_name
14165697:($
"
_user_specified_name
14165695:($
"
_user_specified_name
14165662:($
"
_user_specified_name
14165660:($
"
_user_specified_name
14165642:($
"
_user_specified_name
14165640:($
"
_user_specified_name
14165621:($
"
_user_specified_name
14165619:(
$
"
_user_specified_name
14165616:(	$
"
_user_specified_name
14165614:($
"
_user_specified_name
14165611:($
"
_user_specified_name
14165609:($
"
_user_specified_name
14165606:($
"
_user_specified_name
14165604:($
"
_user_specified_name
14165601:($
"
_user_specified_name
14165599:($
"
_user_specified_name
14165582:($
"
_user_specified_name
14165580:O K
"
_output_shapes
:
%
_user_specified_nameinput_layer
�
�
2__inference_sequential_2360_layer_call_fn_14165212
dense_3041_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3041_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165194s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165208:($
"
_user_specified_name
14165206:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3041_input
�
�
H__inference_dense_3040_layer_call_and_return_conditional_losses_14165579

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
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
:	�d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Q
SoftmaxSoftmaxBiasAdd:output:0*
T0*"
_output_shapes
:[
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*"
_output_shapes
:V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
e
I__inference_flatten_170_layer_call_and_return_conditional_losses_14166059

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
�
�
+__inference_topk_moe_layer_call_fn_14165814
input_layer
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	�

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *O
fJRH
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165701f
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165810:($
"
_user_specified_name
14165808:($
"
_user_specified_name
14165806:($
"
_user_specified_name
14165804:($
"
_user_specified_name
14165802:($
"
_user_specified_name
14165800:($
"
_user_specified_name
14165798:($
"
_user_specified_name
14165796:(
$
"
_user_specified_name
14165794:(	$
"
_user_specified_name
14165792:($
"
_user_specified_name
14165790:($
"
_user_specified_name
14165788:($
"
_user_specified_name
14165786:($
"
_user_specified_name
14165784:($
"
_user_specified_name
14165782:($
"
_user_specified_name
14165780:($
"
_user_specified_name
14165778:($
"
_user_specified_name
14165776:O K
"
_output_shapes
:
%
_user_specified_nameinput_layer
�
�
2__inference_sequential_2372_layer_call_fn_14165378
dense_3053_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3053_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165360s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165374:($
"
_user_specified_name
14165372:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3053_input
�
�
H__inference_dense_3065_layer_call_and_return_conditional_losses_14165514

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_sequential_2360_layer_call_fn_14165221
dense_3041_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3041_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165203s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165217:($
"
_user_specified_name
14165215:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3041_input
�	
�
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165194
dense_3041_input%
dense_3041_14165188:!
dense_3041_14165190:
identity��"dense_3041/StatefulPartitionedCall�
"dense_3041/StatefulPartitionedCallStatefulPartitionedCalldense_3041_inputdense_3041_14165188dense_3041_14165190*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3041_layer_call_and_return_conditional_losses_14165182~
IdentityIdentity+dense_3041/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3041/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3041/StatefulPartitionedCall"dense_3041/StatefulPartitionedCall:($
"
_user_specified_name
14165190:($
"
_user_specified_name
14165188:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3041_input
�
�
H__inference_dense_3053_layer_call_and_return_conditional_losses_14165348

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_dense_3053_layer_call_fn_14166167

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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3053_layer_call_and_return_conditional_losses_14165348s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14166163:($
"
_user_specified_name
14166161:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_topk_moe_layer_call_fn_14165855
input_layer
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:	�

unknown_16:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8� *O
fJRH
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165773f
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165851:($
"
_user_specified_name
14165849:($
"
_user_specified_name
14165847:($
"
_user_specified_name
14165845:($
"
_user_specified_name
14165843:($
"
_user_specified_name
14165841:($
"
_user_specified_name
14165839:($
"
_user_specified_name
14165837:(
$
"
_user_specified_name
14165835:(	$
"
_user_specified_name
14165833:($
"
_user_specified_name
14165831:($
"
_user_specified_name
14165829:($
"
_user_specified_name
14165827:($
"
_user_specified_name
14165825:($
"
_user_specified_name
14165823:($
"
_user_specified_name
14165821:($
"
_user_specified_name
14165819:($
"
_user_specified_name
14165817:O K
"
_output_shapes
:
%
_user_specified_nameinput_layer
�
�
2__inference_sequential_2384_layer_call_fn_14165544
dense_3065_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3065_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165526s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165540:($
"
_user_specified_name
14165538:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3065_input
�
�
H__inference_dense_3065_layer_call_and_return_conditional_losses_14166278

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
H__inference_dense_3047_layer_call_and_return_conditional_losses_14166158

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
ţ
�'
$__inference__traced_restore_14166885
file_prefix4
"assignvariableop_dense_3040_kernel:0
"assignvariableop_1_dense_3040_bias:6
$assignvariableop_2_dense_3066_kernel:0
"assignvariableop_3_dense_3066_bias:6
$assignvariableop_4_dense_3067_kernel:0
"assignvariableop_5_dense_3067_bias:7
$assignvariableop_6_dense_3068_kernel:	�0
"assignvariableop_7_dense_3068_bias:6
$assignvariableop_8_dense_3041_kernel:0
"assignvariableop_9_dense_3041_bias:7
%assignvariableop_10_dense_3047_kernel:1
#assignvariableop_11_dense_3047_bias:7
%assignvariableop_12_dense_3053_kernel:1
#assignvariableop_13_dense_3053_bias:7
%assignvariableop_14_dense_3059_kernel:1
#assignvariableop_15_dense_3059_bias:7
%assignvariableop_16_dense_3065_kernel:1
#assignvariableop_17_dense_3065_bias:'
assignvariableop_18_iteration:	 +
!assignvariableop_19_learning_rate: >
,assignvariableop_20_adam_m_dense_3040_kernel:>
,assignvariableop_21_adam_v_dense_3040_kernel:8
*assignvariableop_22_adam_m_dense_3040_bias:8
*assignvariableop_23_adam_v_dense_3040_bias:>
,assignvariableop_24_adam_m_dense_3041_kernel:>
,assignvariableop_25_adam_v_dense_3041_kernel:8
*assignvariableop_26_adam_m_dense_3041_bias:8
*assignvariableop_27_adam_v_dense_3041_bias:>
,assignvariableop_28_adam_m_dense_3047_kernel:>
,assignvariableop_29_adam_v_dense_3047_kernel:8
*assignvariableop_30_adam_m_dense_3047_bias:8
*assignvariableop_31_adam_v_dense_3047_bias:>
,assignvariableop_32_adam_m_dense_3053_kernel:>
,assignvariableop_33_adam_v_dense_3053_kernel:8
*assignvariableop_34_adam_m_dense_3053_bias:8
*assignvariableop_35_adam_v_dense_3053_bias:>
,assignvariableop_36_adam_m_dense_3059_kernel:>
,assignvariableop_37_adam_v_dense_3059_kernel:8
*assignvariableop_38_adam_m_dense_3059_bias:8
*assignvariableop_39_adam_v_dense_3059_bias:>
,assignvariableop_40_adam_m_dense_3065_kernel:>
,assignvariableop_41_adam_v_dense_3065_kernel:8
*assignvariableop_42_adam_m_dense_3065_bias:8
*assignvariableop_43_adam_v_dense_3065_bias:>
,assignvariableop_44_adam_m_dense_3066_kernel:>
,assignvariableop_45_adam_v_dense_3066_kernel:8
*assignvariableop_46_adam_m_dense_3066_bias:8
*assignvariableop_47_adam_v_dense_3066_bias:>
,assignvariableop_48_adam_m_dense_3067_kernel:>
,assignvariableop_49_adam_v_dense_3067_kernel:8
*assignvariableop_50_adam_m_dense_3067_bias:8
*assignvariableop_51_adam_v_dense_3067_bias:?
,assignvariableop_52_adam_m_dense_3068_kernel:	�?
,assignvariableop_53_adam_v_dense_3068_kernel:	�8
*assignvariableop_54_adam_m_dense_3068_bias:8
*assignvariableop_55_adam_v_dense_3068_bias:%
assignvariableop_56_total_3: %
assignvariableop_57_count_3: %
assignvariableop_58_total_2: %
assignvariableop_59_count_2: %
assignvariableop_60_total_1: %
assignvariableop_61_count_1: #
assignvariableop_62_total: #
assignvariableop_63_count: 
identity_65��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp"assignvariableop_dense_3040_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_3040_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_3066_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_3066_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_3067_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_3067_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp$assignvariableop_6_dense_3068_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_3068_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_dense_3041_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_3041_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp%assignvariableop_10_dense_3047_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_3047_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp%assignvariableop_12_dense_3053_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_3053_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp%assignvariableop_14_dense_3059_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_dense_3059_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp%assignvariableop_16_dense_3065_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp#assignvariableop_17_dense_3065_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_iterationIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_m_dense_3040_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_v_dense_3040_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_m_dense_3040_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_v_dense_3040_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_m_dense_3041_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_v_dense_3041_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_m_dense_3041_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_v_dense_3041_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_m_dense_3047_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp,assignvariableop_29_adam_v_dense_3047_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_m_dense_3047_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_v_dense_3047_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_m_dense_3053_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_v_dense_3053_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_m_dense_3053_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_v_dense_3053_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_m_dense_3059_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp,assignvariableop_37_adam_v_dense_3059_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_m_dense_3059_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_v_dense_3059_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp,assignvariableop_40_adam_m_dense_3065_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_v_dense_3065_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_m_dense_3065_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_v_dense_3065_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_m_dense_3066_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_v_dense_3066_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_m_dense_3066_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_v_dense_3066_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp,assignvariableop_48_adam_m_dense_3067_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_v_dense_3067_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_m_dense_3067_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_v_dense_3067_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp,assignvariableop_52_adam_m_dense_3068_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp,assignvariableop_53_adam_v_dense_3068_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_m_dense_3068_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_v_dense_3068_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_total_3Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpassignvariableop_57_count_3Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpassignvariableop_58_total_2Identity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpassignvariableop_59_count_2Identity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpassignvariableop_60_total_1Identity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpassignvariableop_61_count_1Identity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpassignvariableop_62_totalIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpassignvariableop_63_countIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_65IdentityIdentity_64:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_65Identity_65:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%@!

_user_specified_namecount:%?!

_user_specified_nametotal:'>#
!
_user_specified_name	count_1:'=#
!
_user_specified_name	total_1:'<#
!
_user_specified_name	count_2:';#
!
_user_specified_name	total_2:':#
!
_user_specified_name	count_3:'9#
!
_user_specified_name	total_3:682
0
_user_specified_nameAdam/v/dense_3068/bias:672
0
_user_specified_nameAdam/m/dense_3068/bias:864
2
_user_specified_nameAdam/v/dense_3068/kernel:854
2
_user_specified_nameAdam/m/dense_3068/kernel:642
0
_user_specified_nameAdam/v/dense_3067/bias:632
0
_user_specified_nameAdam/m/dense_3067/bias:824
2
_user_specified_nameAdam/v/dense_3067/kernel:814
2
_user_specified_nameAdam/m/dense_3067/kernel:602
0
_user_specified_nameAdam/v/dense_3066/bias:6/2
0
_user_specified_nameAdam/m/dense_3066/bias:8.4
2
_user_specified_nameAdam/v/dense_3066/kernel:8-4
2
_user_specified_nameAdam/m/dense_3066/kernel:6,2
0
_user_specified_nameAdam/v/dense_3065/bias:6+2
0
_user_specified_nameAdam/m/dense_3065/bias:8*4
2
_user_specified_nameAdam/v/dense_3065/kernel:8)4
2
_user_specified_nameAdam/m/dense_3065/kernel:6(2
0
_user_specified_nameAdam/v/dense_3059/bias:6'2
0
_user_specified_nameAdam/m/dense_3059/bias:8&4
2
_user_specified_nameAdam/v/dense_3059/kernel:8%4
2
_user_specified_nameAdam/m/dense_3059/kernel:6$2
0
_user_specified_nameAdam/v/dense_3053/bias:6#2
0
_user_specified_nameAdam/m/dense_3053/bias:8"4
2
_user_specified_nameAdam/v/dense_3053/kernel:8!4
2
_user_specified_nameAdam/m/dense_3053/kernel:6 2
0
_user_specified_nameAdam/v/dense_3047/bias:62
0
_user_specified_nameAdam/m/dense_3047/bias:84
2
_user_specified_nameAdam/v/dense_3047/kernel:84
2
_user_specified_nameAdam/m/dense_3047/kernel:62
0
_user_specified_nameAdam/v/dense_3041/bias:62
0
_user_specified_nameAdam/m/dense_3041/bias:84
2
_user_specified_nameAdam/v/dense_3041/kernel:84
2
_user_specified_nameAdam/m/dense_3041/kernel:62
0
_user_specified_nameAdam/v/dense_3040/bias:62
0
_user_specified_nameAdam/m/dense_3040/bias:84
2
_user_specified_nameAdam/v/dense_3040/kernel:84
2
_user_specified_nameAdam/m/dense_3040/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:/+
)
_user_specified_namedense_3065/bias:1-
+
_user_specified_namedense_3065/kernel:/+
)
_user_specified_namedense_3059/bias:1-
+
_user_specified_namedense_3059/kernel:/+
)
_user_specified_namedense_3053/bias:1-
+
_user_specified_namedense_3053/kernel:/+
)
_user_specified_namedense_3047/bias:1-
+
_user_specified_namedense_3047/kernel:/
+
)
_user_specified_namedense_3041/bias:1	-
+
_user_specified_namedense_3041/kernel:/+
)
_user_specified_namedense_3068/bias:1-
+
_user_specified_namedense_3068/kernel:/+
)
_user_specified_namedense_3067/bias:1-
+
_user_specified_namedense_3067/kernel:/+
)
_user_specified_namedense_3066/bias:1-
+
_user_specified_namedense_3066/kernel:/+
)
_user_specified_namedense_3040/bias:1-
+
_user_specified_namedense_3040/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
H__inference_dense_3059_layer_call_and_return_conditional_losses_14166238

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_flatten_170_layer_call_and_return_conditional_losses_14165683

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
�

h
I__inference_dropout_170_layer_call_and_return_conditional_losses_14165676

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
�
�
H__inference_dense_3040_layer_call_and_return_conditional_losses_14165974

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
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
:	�d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Q
SoftmaxSoftmaxBiasAdd:output:0*
T0*"
_output_shapes
:[
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*"
_output_shapes
:V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
2__inference_sequential_2378_layer_call_fn_14165461
dense_3059_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3059_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165443s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165457:($
"
_user_specified_name
14165455:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3059_input
�
�
H__inference_dense_3067_layer_call_and_return_conditional_losses_14165659

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"�     p
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	��
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	�d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:K
ReluReluBiasAdd:output:0*
T0*"
_output_shapes
:\
IdentityIdentityRelu:activations:0^NoOp*
T0*"
_output_shapes
:V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
-__inference_dense_3068_layer_call_fn_14166068

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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3068_layer_call_and_return_conditional_losses_14165694f
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14166064:($
"
_user_specified_name
14166062:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�	
�
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165360
dense_3053_input%
dense_3053_14165354:!
dense_3053_14165356:
identity��"dense_3053/StatefulPartitionedCall�
"dense_3053/StatefulPartitionedCallStatefulPartitionedCalldense_3053_inputdense_3053_14165354dense_3053_14165356*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3053_layer_call_and_return_conditional_losses_14165348~
IdentityIdentity+dense_3053/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3053/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3053/StatefulPartitionedCall"dense_3053/StatefulPartitionedCall:($
"
_user_specified_name
14165356:($
"
_user_specified_name
14165354:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3053_input
�
�
2__inference_sequential_2384_layer_call_fn_14165553
dense_3065_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3065_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165535s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165549:($
"
_user_specified_name
14165547:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3065_input
�
�
H__inference_dense_3041_layer_call_and_return_conditional_losses_14165182

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
H__inference_dense_3068_layer_call_and_return_conditional_losses_14166078

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
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�
g
.__inference_dropout_170_layer_call_fn_14166026

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
GPU 2J 8� *R
fMRK
I__inference_dropout_170_layer_call_and_return_conditional_losses_14165676j
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
�	
�
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165526
dense_3065_input%
dense_3065_14165520:!
dense_3065_14165522:
identity��"dense_3065/StatefulPartitionedCall�
"dense_3065/StatefulPartitionedCallStatefulPartitionedCalldense_3065_inputdense_3065_14165520dense_3065_14165522*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3065_layer_call_and_return_conditional_losses_14165514~
IdentityIdentity+dense_3065/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3065/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3065/StatefulPartitionedCall"dense_3065/StatefulPartitionedCall:($
"
_user_specified_name
14165522:($
"
_user_specified_name
14165520:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3065_input
�	
�
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165443
dense_3059_input%
dense_3059_14165437:!
dense_3059_14165439:
identity��"dense_3059/StatefulPartitionedCall�
"dense_3059/StatefulPartitionedCallStatefulPartitionedCalldense_3059_inputdense_3059_14165437dense_3059_14165439*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3059_layer_call_and_return_conditional_losses_14165431~
IdentityIdentity+dense_3059/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3059/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3059/StatefulPartitionedCall"dense_3059/StatefulPartitionedCall:($
"
_user_specified_name
14165439:($
"
_user_specified_name
14165437:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3059_input
�
�
H__inference_dense_3066_layer_call_and_return_conditional_losses_14165997

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
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
:	�d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*"
_output_shapes
:V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
-__inference_dense_3040_layer_call_fn_14165959

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_3040_layer_call_and_return_conditional_losses_14165579j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165955:($
"
_user_specified_name
14165953:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�	
�
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165277
dense_3047_input%
dense_3047_14165271:!
dense_3047_14165273:
identity��"dense_3047/StatefulPartitionedCall�
"dense_3047/StatefulPartitionedCallStatefulPartitionedCalldense_3047_inputdense_3047_14165271dense_3047_14165273*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3047_layer_call_and_return_conditional_losses_14165265~
IdentityIdentity+dense_3047/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3047/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3047/StatefulPartitionedCall"dense_3047/StatefulPartitionedCall:($
"
_user_specified_name
14165273:($
"
_user_specified_name
14165271:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3047_input
�
�
H__inference_dense_3059_layer_call_and_return_conditional_losses_14165431

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
J
.__inference_flatten_170_layer_call_fn_14166053

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
GPU 2J 8� *R
fMRK
I__inference_flatten_170_layer_call_and_return_conditional_losses_14165683X
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
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165452
dense_3059_input%
dense_3059_14165446:!
dense_3059_14165448:
identity��"dense_3059/StatefulPartitionedCall�
"dense_3059/StatefulPartitionedCallStatefulPartitionedCalldense_3059_inputdense_3059_14165446dense_3059_14165448*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3059_layer_call_and_return_conditional_losses_14165431~
IdentityIdentity+dense_3059/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3059/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3059/StatefulPartitionedCall"dense_3059/StatefulPartitionedCall:($
"
_user_specified_name
14165448:($
"
_user_specified_name
14165446:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3059_input
�
�
2__inference_sequential_2366_layer_call_fn_14165295
dense_3047_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3047_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165277s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165291:($
"
_user_specified_name
14165289:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3047_input
�	
�
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165369
dense_3053_input%
dense_3053_14165363:!
dense_3053_14165365:
identity��"dense_3053/StatefulPartitionedCall�
"dense_3053/StatefulPartitionedCallStatefulPartitionedCalldense_3053_inputdense_3053_14165363dense_3053_14165365*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3053_layer_call_and_return_conditional_losses_14165348~
IdentityIdentity+dense_3053/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3053/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3053/StatefulPartitionedCall"dense_3053/StatefulPartitionedCall:($
"
_user_specified_name
14165365:($
"
_user_specified_name
14165363:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3053_input
�
�
2__inference_sequential_2366_layer_call_fn_14165304
dense_3047_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3047_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165286s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165300:($
"
_user_specified_name
14165298:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3047_input
�	
�
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165203
dense_3041_input%
dense_3041_14165197:!
dense_3041_14165199:
identity��"dense_3041/StatefulPartitionedCall�
"dense_3041/StatefulPartitionedCallStatefulPartitionedCalldense_3041_inputdense_3041_14165197dense_3041_14165199*
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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3041_layer_call_and_return_conditional_losses_14165182~
IdentityIdentity+dense_3041/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������G
NoOpNoOp#^dense_3041/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 2H
"dense_3041/StatefulPartitionedCall"dense_3041/StatefulPartitionedCall:($
"
_user_specified_name
14165199:($
"
_user_specified_name
14165197:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3041_input
�
�
-__inference_dense_3067_layer_call_fn_14166006

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_dense_3067_layer_call_and_return_conditional_losses_14165659j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14166002:($
"
_user_specified_name
14166000:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
g
I__inference_dropout_170_layer_call_and_return_conditional_losses_14165764

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
�
�
H__inference_dense_3066_layer_call_and_return_conditional_losses_14165639

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
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
:	�d
Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         w
	TensordotReshapeTensordot/MatMul:product:0Tensordot/shape:output:0*
T0*"
_output_shapes
:r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0s
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*"
_output_shapes
:V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:J F
"
_output_shapes
:
 
_user_specified_nameinputs
�
�
-__inference_dense_3065_layer_call_fn_14166247

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
GPU 2J 8� *Q
fLRJ
H__inference_dense_3065_layer_call_and_return_conditional_losses_14165514s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14166243:($
"
_user_specified_name
14166241:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_sequential_2378_layer_call_fn_14165470
dense_3059_input
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_3059_inputunknown	unknown_0*
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
GPU 2J 8� *V
fQRO
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165452s
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
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
14165466:($
"
_user_specified_name
14165464:] Y
+
_output_shapes
:���������
*
_user_specified_namedense_3059_input
�
g
I__inference_dropout_170_layer_call_and_return_conditional_losses_14166048

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
�
�
H__inference_dense_3047_layer_call_and_return_conditional_losses_14165265

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
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
>
input_layer/
serving_default_input_layer:05

dense_3068'
StatefulPartitionedCall:0tensorflow/serving/predict:��
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
layer_with_weights-5
layer-11
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer-17
layer_with_weights-8
layer-18
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
(
%	keras_api"
_tf_keras_layer
(
&	keras_api"
_tf_keras_layer
(
'	keras_api"
_tf_keras_layer
(
(	keras_api"
_tf_keras_layer
(
)	keras_api"
_tf_keras_layer
�
*layer_with_weights-0
*layer-0
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
1layer_with_weights-0
1layer-0
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
8layer_with_weights-0
8layer-0
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
?layer_with_weights-0
?layer-0
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
Flayer_with_weights-0
Flayer-0
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_sequential
(
M	keras_api"
_tf_keras_layer
(
N	keras_api"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses

]kernel
^bias"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e_random_generator"
_tf_keras_layer
�
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias"
_tf_keras_layer
�
#0
$1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11
U12
V13
]14
^15
r16
s17"
trackable_list_wrapper
�
#0
$1
t2
u3
v4
w5
x6
y7
z8
{9
|10
}11
U12
V13
]14
^15
r16
s17"
trackable_list_wrapper
 "
trackable_list_wrapper
�
~non_trainable_variables

layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_topk_moe_layer_call_fn_14165814
+__inference_topk_moe_layer_call_fn_14165855�
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
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165701
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165773�
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
#__inference__wrapped_model_14165147input_layer"�
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
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_dense_3040_layer_call_fn_14165959�
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
H__inference_dense_3040_layer_call_and_return_conditional_losses_14165974�
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
#:!2dense_3040/kernel
:2dense_3040/bias
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

tkernel
ubias"
_tf_keras_layer
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_sequential_2360_layer_call_fn_14165212
2__inference_sequential_2360_layer_call_fn_14165221�
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
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165194
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165203�
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

vkernel
wbias"
_tf_keras_layer
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_sequential_2366_layer_call_fn_14165295
2__inference_sequential_2366_layer_call_fn_14165304�
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
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165277
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165286�
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

xkernel
ybias"
_tf_keras_layer
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_sequential_2372_layer_call_fn_14165378
2__inference_sequential_2372_layer_call_fn_14165387�
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
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165360
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165369�
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

zkernel
{bias"
_tf_keras_layer
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_sequential_2378_layer_call_fn_14165461
2__inference_sequential_2378_layer_call_fn_14165470�
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
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165443
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165452�
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
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

|kernel
}bias"
_tf_keras_layer
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
2__inference_sequential_2384_layer_call_fn_14165544
2__inference_sequential_2384_layer_call_fn_14165553�
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
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165526
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165535�
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
_generic_user_object
"
_generic_user_object
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_dense_3066_layer_call_fn_14165983�
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
H__inference_dense_3066_layer_call_and_return_conditional_losses_14165997�
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
#:!2dense_3066/kernel
:2dense_3066/bias
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_dense_3067_layer_call_fn_14166006�
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
H__inference_dense_3067_layer_call_and_return_conditional_losses_14166021�
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
#:!2dense_3067/kernel
:2dense_3067/bias
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
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_dropout_170_layer_call_fn_14166026
.__inference_dropout_170_layer_call_fn_14166031�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_dropout_170_layer_call_and_return_conditional_losses_14166043
I__inference_dropout_170_layer_call_and_return_conditional_losses_14166048�
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
 z�trace_0z�trace_1
"
_generic_user_object
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
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_flatten_170_layer_call_fn_14166053�
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
I__inference_flatten_170_layer_call_and_return_conditional_losses_14166059�
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
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_dense_3068_layer_call_fn_14166068�
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
H__inference_dense_3068_layer_call_and_return_conditional_losses_14166078�
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
$:"	�2dense_3068/kernel
:2dense_3068/bias
#:!2dense_3041/kernel
:2dense_3041/bias
#:!2dense_3047/kernel
:2dense_3047/bias
#:!2dense_3053/kernel
:2dense_3053/bias
#:!2dense_3059/kernel
:2dense_3059/bias
#:!2dense_3065/kernel
:2dense_3065/bias
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
16
17
18"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_topk_moe_layer_call_fn_14165814input_layer"�
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
+__inference_topk_moe_layer_call_fn_14165855input_layer"�
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
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165701input_layer"�
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
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165773input_layer"�
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
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
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
 0
�B�
&__inference_signature_wrapper_14165950input_layer"�
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
-__inference_dense_3040_layer_call_fn_14165959inputs"�
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
H__inference_dense_3040_layer_call_and_return_conditional_losses_14165974inputs"�
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
.
t0
u1"
trackable_list_wrapper
.
t0
u1"
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
-__inference_dense_3041_layer_call_fn_14166087�
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
H__inference_dense_3041_layer_call_and_return_conditional_losses_14166118�
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
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_sequential_2360_layer_call_fn_14165212dense_3041_input"�
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
2__inference_sequential_2360_layer_call_fn_14165221dense_3041_input"�
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
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165194dense_3041_input"�
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
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165203dense_3041_input"�
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
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
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
-__inference_dense_3047_layer_call_fn_14166127�
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
H__inference_dense_3047_layer_call_and_return_conditional_losses_14166158�
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
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_sequential_2366_layer_call_fn_14165295dense_3047_input"�
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
2__inference_sequential_2366_layer_call_fn_14165304dense_3047_input"�
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
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165277dense_3047_input"�
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
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165286dense_3047_input"�
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
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
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
-__inference_dense_3053_layer_call_fn_14166167�
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
H__inference_dense_3053_layer_call_and_return_conditional_losses_14166198�
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
trackable_list_wrapper
'
80"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_sequential_2372_layer_call_fn_14165378dense_3053_input"�
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
2__inference_sequential_2372_layer_call_fn_14165387dense_3053_input"�
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
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165360dense_3053_input"�
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
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165369dense_3053_input"�
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
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
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
-__inference_dense_3059_layer_call_fn_14166207�
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
H__inference_dense_3059_layer_call_and_return_conditional_losses_14166238�
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
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_sequential_2378_layer_call_fn_14165461dense_3059_input"�
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
2__inference_sequential_2378_layer_call_fn_14165470dense_3059_input"�
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
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165443dense_3059_input"�
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
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165452dense_3059_input"�
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
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
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
-__inference_dense_3065_layer_call_fn_14166247�
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
H__inference_dense_3065_layer_call_and_return_conditional_losses_14166278�
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
trackable_list_wrapper
'
F0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
2__inference_sequential_2384_layer_call_fn_14165544dense_3065_input"�
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
2__inference_sequential_2384_layer_call_fn_14165553dense_3065_input"�
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
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165526dense_3065_input"�
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
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165535dense_3065_input"�
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
�B�
-__inference_dense_3066_layer_call_fn_14165983inputs"�
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
H__inference_dense_3066_layer_call_and_return_conditional_losses_14165997inputs"�
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
-__inference_dense_3067_layer_call_fn_14166006inputs"�
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
H__inference_dense_3067_layer_call_and_return_conditional_losses_14166021inputs"�
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
.__inference_dropout_170_layer_call_fn_14166026inputs"�
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
.__inference_dropout_170_layer_call_fn_14166031inputs"�
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
I__inference_dropout_170_layer_call_and_return_conditional_losses_14166043inputs"�
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
I__inference_dropout_170_layer_call_and_return_conditional_losses_14166048inputs"�
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
.__inference_flatten_170_layer_call_fn_14166053inputs"�
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
I__inference_flatten_170_layer_call_and_return_conditional_losses_14166059inputs"�
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
-__inference_dense_3068_layer_call_fn_14166068inputs"�
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
H__inference_dense_3068_layer_call_and_return_conditional_losses_14166078inputs"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
(:&2Adam/m/dense_3040/kernel
(:&2Adam/v/dense_3040/kernel
": 2Adam/m/dense_3040/bias
": 2Adam/v/dense_3040/bias
(:&2Adam/m/dense_3041/kernel
(:&2Adam/v/dense_3041/kernel
": 2Adam/m/dense_3041/bias
": 2Adam/v/dense_3041/bias
(:&2Adam/m/dense_3047/kernel
(:&2Adam/v/dense_3047/kernel
": 2Adam/m/dense_3047/bias
": 2Adam/v/dense_3047/bias
(:&2Adam/m/dense_3053/kernel
(:&2Adam/v/dense_3053/kernel
": 2Adam/m/dense_3053/bias
": 2Adam/v/dense_3053/bias
(:&2Adam/m/dense_3059/kernel
(:&2Adam/v/dense_3059/kernel
": 2Adam/m/dense_3059/bias
": 2Adam/v/dense_3059/bias
(:&2Adam/m/dense_3065/kernel
(:&2Adam/v/dense_3065/kernel
": 2Adam/m/dense_3065/bias
": 2Adam/v/dense_3065/bias
(:&2Adam/m/dense_3066/kernel
(:&2Adam/v/dense_3066/kernel
": 2Adam/m/dense_3066/bias
": 2Adam/v/dense_3066/bias
(:&2Adam/m/dense_3067/kernel
(:&2Adam/v/dense_3067/kernel
": 2Adam/m/dense_3067/bias
": 2Adam/v/dense_3067/bias
):'	�2Adam/m/dense_3068/kernel
):'	�2Adam/v/dense_3068/kernel
": 2Adam/m/dense_3068/bias
": 2Adam/v/dense_3068/bias
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
-__inference_dense_3041_layer_call_fn_14166087inputs"�
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
H__inference_dense_3041_layer_call_and_return_conditional_losses_14166118inputs"�
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
-__inference_dense_3047_layer_call_fn_14166127inputs"�
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
H__inference_dense_3047_layer_call_and_return_conditional_losses_14166158inputs"�
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
-__inference_dense_3053_layer_call_fn_14166167inputs"�
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
H__inference_dense_3053_layer_call_and_return_conditional_losses_14166198inputs"�
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
-__inference_dense_3059_layer_call_fn_14166207inputs"�
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
H__inference_dense_3059_layer_call_and_return_conditional_losses_14166238inputs"�
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
-__inference_dense_3065_layer_call_fn_14166247inputs"�
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
H__inference_dense_3065_layer_call_and_return_conditional_losses_14166278inputs"�
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
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
#__inference__wrapped_model_14165147u#$tuvwxyz{|}UV]^rs/�,
%�"
 �
input_layer
� ".�+
)

dense_3068�

dense_3068�
H__inference_dense_3040_layer_call_and_return_conditional_losses_14165974Y#$*�'
 �
�
inputs
� "'�$
�
tensor_0
� 
-__inference_dense_3040_layer_call_fn_14165959N#$*�'
 �
�
inputs
� "�
unknown�
H__inference_dense_3041_layer_call_and_return_conditional_losses_14166118ktu3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
-__inference_dense_3041_layer_call_fn_14166087`tu3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
H__inference_dense_3047_layer_call_and_return_conditional_losses_14166158kvw3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
-__inference_dense_3047_layer_call_fn_14166127`vw3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
H__inference_dense_3053_layer_call_and_return_conditional_losses_14166198kxy3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
-__inference_dense_3053_layer_call_fn_14166167`xy3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
H__inference_dense_3059_layer_call_and_return_conditional_losses_14166238kz{3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
-__inference_dense_3059_layer_call_fn_14166207`z{3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
H__inference_dense_3065_layer_call_and_return_conditional_losses_14166278k|}3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������
� �
-__inference_dense_3065_layer_call_fn_14166247`|}3�0
)�&
$�!
inputs���������
� "%�"
unknown����������
H__inference_dense_3066_layer_call_and_return_conditional_losses_14165997YUV*�'
 �
�
inputs
� "'�$
�
tensor_0
� 
-__inference_dense_3066_layer_call_fn_14165983NUV*�'
 �
�
inputs
� "�
unknown�
H__inference_dense_3067_layer_call_and_return_conditional_losses_14166021Y]^*�'
 �
�
inputs
� "'�$
�
tensor_0
� 
-__inference_dense_3067_layer_call_fn_14166006N]^*�'
 �
�
inputs
� "�
unknown�
H__inference_dense_3068_layer_call_and_return_conditional_losses_14166078Rrs'�$
�
�
inputs	�
� "#� 
�
tensor_0
� x
-__inference_dense_3068_layer_call_fn_14166068Grs'�$
�
�
inputs	�
� "�
unknown�
I__inference_dropout_170_layer_call_and_return_conditional_losses_14166043Y.�+
$�!
�
inputs
p
� "'�$
�
tensor_0
� �
I__inference_dropout_170_layer_call_and_return_conditional_losses_14166048Y.�+
$�!
�
inputs
p 
� "'�$
�
tensor_0
� �
.__inference_dropout_170_layer_call_fn_14166026N.�+
$�!
�
inputs
p
� "�
unknown�
.__inference_dropout_170_layer_call_fn_14166031N.�+
$�!
�
inputs
p 
� "�
unknown�
I__inference_flatten_170_layer_call_and_return_conditional_losses_14166059R*�'
 �
�
inputs
� "$�!
�
tensor_0	�
� y
.__inference_flatten_170_layer_call_fn_14166053G*�'
 �
�
inputs
� "�
unknown	��
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165194}tuE�B
;�8
.�+
dense_3041_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
M__inference_sequential_2360_layer_call_and_return_conditional_losses_14165203}tuE�B
;�8
.�+
dense_3041_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
2__inference_sequential_2360_layer_call_fn_14165212rtuE�B
;�8
.�+
dense_3041_input���������
p

 
� "%�"
unknown����������
2__inference_sequential_2360_layer_call_fn_14165221rtuE�B
;�8
.�+
dense_3041_input���������
p 

 
� "%�"
unknown����������
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165277}vwE�B
;�8
.�+
dense_3047_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
M__inference_sequential_2366_layer_call_and_return_conditional_losses_14165286}vwE�B
;�8
.�+
dense_3047_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
2__inference_sequential_2366_layer_call_fn_14165295rvwE�B
;�8
.�+
dense_3047_input���������
p

 
� "%�"
unknown����������
2__inference_sequential_2366_layer_call_fn_14165304rvwE�B
;�8
.�+
dense_3047_input���������
p 

 
� "%�"
unknown����������
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165360}xyE�B
;�8
.�+
dense_3053_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
M__inference_sequential_2372_layer_call_and_return_conditional_losses_14165369}xyE�B
;�8
.�+
dense_3053_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
2__inference_sequential_2372_layer_call_fn_14165378rxyE�B
;�8
.�+
dense_3053_input���������
p

 
� "%�"
unknown����������
2__inference_sequential_2372_layer_call_fn_14165387rxyE�B
;�8
.�+
dense_3053_input���������
p 

 
� "%�"
unknown����������
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165443}z{E�B
;�8
.�+
dense_3059_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
M__inference_sequential_2378_layer_call_and_return_conditional_losses_14165452}z{E�B
;�8
.�+
dense_3059_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
2__inference_sequential_2378_layer_call_fn_14165461rz{E�B
;�8
.�+
dense_3059_input���������
p

 
� "%�"
unknown����������
2__inference_sequential_2378_layer_call_fn_14165470rz{E�B
;�8
.�+
dense_3059_input���������
p 

 
� "%�"
unknown����������
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165526}|}E�B
;�8
.�+
dense_3065_input���������
p

 
� "0�-
&�#
tensor_0���������
� �
M__inference_sequential_2384_layer_call_and_return_conditional_losses_14165535}|}E�B
;�8
.�+
dense_3065_input���������
p 

 
� "0�-
&�#
tensor_0���������
� �
2__inference_sequential_2384_layer_call_fn_14165544r|}E�B
;�8
.�+
dense_3065_input���������
p

 
� "%�"
unknown����������
2__inference_sequential_2384_layer_call_fn_14165553r|}E�B
;�8
.�+
dense_3065_input���������
p 

 
� "%�"
unknown����������
&__inference_signature_wrapper_14165950�#$tuvwxyz{|}UV]^rs>�;
� 
4�1
/
input_layer �
input_layer".�+
)

dense_3068�

dense_3068�
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165701r#$tuvwxyz{|}UV]^rs7�4
-�*
 �
input_layer
p

 
� "#� 
�
tensor_0
� �
F__inference_topk_moe_layer_call_and_return_conditional_losses_14165773r#$tuvwxyz{|}UV]^rs7�4
-�*
 �
input_layer
p 

 
� "#� 
�
tensor_0
� �
+__inference_topk_moe_layer_call_fn_14165814g#$tuvwxyz{|}UV]^rs7�4
-�*
 �
input_layer
p

 
� "�
unknown�
+__inference_topk_moe_layer_call_fn_14165855g#$tuvwxyz{|}UV]^rs7�4
-�*
 �
input_layer
p 

 
� "�
unknown