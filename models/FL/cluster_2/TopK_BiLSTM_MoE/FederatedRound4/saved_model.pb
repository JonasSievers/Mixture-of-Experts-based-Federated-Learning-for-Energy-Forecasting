│Ѓ9
» 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
ђ
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
resourceѕ
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
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
Ї
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint         "	
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
dtypetypeѕ
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
list(type)(0ѕ
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
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
э
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
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
г
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
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
ѕ"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628§Ѕ4
И
0bidirectional_92/backward_lstm_92/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20bidirectional_92/backward_lstm_92/lstm_cell/bias
▒
Dbidirectional_92/backward_lstm_92/lstm_cell/bias/Read/ReadVariableOpReadVariableOp0bidirectional_92/backward_lstm_92/lstm_cell/bias*
_output_shapes
: *
dtype0
н
<bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *M
shared_name><bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel
═
Pbidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp<bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel*
_output_shapes

: *
dtype0
└
2bidirectional_92/backward_lstm_92/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *C
shared_name42bidirectional_92/backward_lstm_92/lstm_cell/kernel
╣
Fbidirectional_92/backward_lstm_92/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp2bidirectional_92/backward_lstm_92/lstm_cell/kernel*
_output_shapes

: *
dtype0
Х
/bidirectional_92/forward_lstm_92/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/bidirectional_92/forward_lstm_92/lstm_cell/bias
»
Cbidirectional_92/forward_lstm_92/lstm_cell/bias/Read/ReadVariableOpReadVariableOp/bidirectional_92/forward_lstm_92/lstm_cell/bias*
_output_shapes
: *
dtype0
м
;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *L
shared_name=;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel
╦
Obidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel*
_output_shapes

: *
dtype0
Й
1bidirectional_92/forward_lstm_92/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *B
shared_name31bidirectional_92/forward_lstm_92/lstm_cell/kernel
и
Ebidirectional_92/forward_lstm_92/lstm_cell/kernel/Read/ReadVariableOpReadVariableOp1bidirectional_92/forward_lstm_92/lstm_cell/kernel*
_output_shapes

: *
dtype0
t
dense_592/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_592/bias
m
"dense_592/bias/Read/ReadVariableOpReadVariableOpdense_592/bias*
_output_shapes
:*
dtype0
|
dense_592/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_592/kernel
u
$dense_592/kernel/Read/ReadVariableOpReadVariableOpdense_592/kernel*
_output_shapes

:*
dtype0
t
dense_587/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_587/bias
m
"dense_587/bias/Read/ReadVariableOpReadVariableOpdense_587/bias*
_output_shapes
:*
dtype0
|
dense_587/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_587/kernel
u
$dense_587/kernel/Read/ReadVariableOpReadVariableOpdense_587/kernel*
_output_shapes

:*
dtype0
t
dense_582/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_582/bias
m
"dense_582/bias/Read/ReadVariableOpReadVariableOpdense_582/bias*
_output_shapes
:*
dtype0
|
dense_582/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_582/kernel
u
$dense_582/kernel/Read/ReadVariableOpReadVariableOpdense_582/kernel*
_output_shapes

:*
dtype0
t
dense_577/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_577/bias
m
"dense_577/bias/Read/ReadVariableOpReadVariableOpdense_577/bias*
_output_shapes
:*
dtype0
|
dense_577/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_577/kernel
u
$dense_577/kernel/Read/ReadVariableOpReadVariableOpdense_577/kernel*
_output_shapes

:*
dtype0
t
dense_593/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_593/bias
m
"dense_593/bias/Read/ReadVariableOpReadVariableOpdense_593/bias*
_output_shapes
:*
dtype0
}
dense_593/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*!
shared_namedense_593/kernel
v
$dense_593/kernel/Read/ReadVariableOpReadVariableOpdense_593/kernel*
_output_shapes
:	ђ*
dtype0
t
dense_576/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_576/bias
m
"dense_576/bias/Read/ReadVariableOpReadVariableOpdense_576/bias*
_output_shapes
:*
dtype0
|
dense_576/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_576/kernel
u
$dense_576/kernel/Read/ReadVariableOpReadVariableOpdense_576/kernel*
_output_shapes

:*
dtype0
p
serving_default_input_3Placeholder*"
_output_shapes
:*
dtype0*
shape:
Я
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3dense_576/kerneldense_576/biasdense_577/kerneldense_577/biasdense_582/kerneldense_582/biasdense_587/kerneldense_587/biasdense_592/kerneldense_592/bias1bidirectional_92/forward_lstm_92/lstm_cell/kernel;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel/bidirectional_92/forward_lstm_92/lstm_cell/bias2bidirectional_92/backward_lstm_92/lstm_cell/kernel<bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel0bidirectional_92/backward_lstm_92/lstm_cell/biasdense_593/kerneldense_593/bias*
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
GPU 2J 8ѓ */
f*R(
&__inference_signature_wrapper_55811219

NoOpNoOp
њЁ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╠ё
value┴ёBйё Bхё
Ч
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
╦
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
│
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
#/_self_saveable_object_factories* 
│
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
▄
9layer_with_weights-0
9layer-0
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses
#@_self_saveable_object_factories*
▄
Alayer_with_weights-0
Alayer-0
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
#H_self_saveable_object_factories*
▄
Ilayer_with_weights-0
Ilayer-0
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
#P_self_saveable_object_factories*
▄
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
│
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
#a_self_saveable_object_factories* 
▄
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
hforward_layer
ibackward_layer
#j_self_saveable_object_factories*
╩
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
q_random_generator
#r_self_saveable_object_factories* 
│
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
#y_self_saveable_object_factories* 
╬
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
ђkernel
	Ђbias
$ѓ_self_saveable_object_factories*
џ
"0
#1
Ѓ2
ё3
Ё4
є5
Є6
ѕ7
Ѕ8
і9
І10
ї11
Ї12
ј13
Ј14
љ15
ђ16
Ђ17*
џ
"0
#1
Ѓ2
ё3
Ё4
є5
Є6
ѕ7
Ѕ8
і9
І10
ї11
Ї12
ј13
Ј14
љ15
ђ16
Ђ17*
* 
х
Љnon_trainable_variables
њlayers
Њmetrics
 ћlayer_regularization_losses
Ћlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ќtrace_0
Ќtrace_1* 

ўtrace_0
Ўtrace_1* 
* 

џserving_default* 
* 
* 

"0
#1*

"0
#1*
* 
ў
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

аtrace_0* 

Аtrace_0* 
`Z
VARIABLE_VALUEdense_576/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_576/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
ќ
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses* 

Дtrace_0
еtrace_1* 

Еtrace_0
фtrace_1* 
* 
* 
* 
* 
ќ
Фnon_trainable_variables
гlayers
Гmetrics
 «layer_regularization_losses
»layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses* 

░trace_0
▒trace_1* 

▓trace_0
│trace_1* 
* 
* 
* 
н
┤	variables
хtrainable_variables
Хregularization_losses
и	keras_api
И__call__
+╣&call_and_return_all_conditional_losses
Ѓkernel
	ёbias
$║_self_saveable_object_factories*

Ѓ0
ё1*

Ѓ0
ё1*
* 
ў
╗non_trainable_variables
╝layers
йmetrics
 Йlayer_regularization_losses
┐layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

└trace_0
┴trace_1* 

┬trace_0
├trace_1* 
* 
н
─	variables
┼trainable_variables
кregularization_losses
К	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses
Ёkernel
	єbias
$╩_self_saveable_object_factories*

Ё0
є1*

Ё0
є1*
* 
ў
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

лtrace_0
Лtrace_1* 

мtrace_0
Мtrace_1* 
* 
н
н	variables
Нtrainable_variables
оregularization_losses
О	keras_api
п__call__
+┘&call_and_return_all_conditional_losses
Єkernel
	ѕbias
$┌_self_saveable_object_factories*

Є0
ѕ1*

Є0
ѕ1*
* 
ў
█non_trainable_variables
▄layers
Пmetrics
 яlayer_regularization_losses
▀layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

Яtrace_0
рtrace_1* 

Рtrace_0
сtrace_1* 
* 
н
С	variables
тtrainable_variables
Тregularization_losses
у	keras_api
У__call__
+ж&call_and_return_all_conditional_losses
Ѕkernel
	іbias
$Ж_self_saveable_object_factories*

Ѕ0
і1*

Ѕ0
і1*
* 
ў
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

­trace_0
ыtrace_1* 

Ыtrace_0
зtrace_1* 
* 
* 
* 
* 
* 
* 
ќ
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses* 

щtrace_0
Щtrace_1* 

чtrace_0
Чtrace_1* 
* 
4
І0
ї1
Ї2
ј3
Ј4
љ5*
4
І0
ї1
Ї2
ј3
Ј4
љ5*
* 
ў
§non_trainable_variables
■layers
 metrics
 ђlayer_regularization_losses
Ђlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
:
ѓtrace_0
Ѓtrace_1
ёtrace_2
Ёtrace_3* 
:
єtrace_0
Єtrace_1
ѕtrace_2
Ѕtrace_3* 
­
і	variables
Іtrainable_variables
їregularization_losses
Ї	keras_api
ј__call__
+Ј&call_and_return_all_conditional_losses
љ_random_generator
	Љcell
њ
state_spec
$Њ_self_saveable_object_factories*
­
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses
џ_random_generator
	Џcell
ю
state_spec
$Ю_self_saveable_object_factories*
* 
* 
* 
* 
ќ
ъnon_trainable_variables
Ъlayers
аmetrics
 Аlayer_regularization_losses
бlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses* 

Бtrace_0
цtrace_1* 

Цtrace_0
дtrace_1* 
(
$Д_self_saveable_object_factories* 
* 
* 
* 
* 
ќ
еnon_trainable_variables
Еlayers
фmetrics
 Фlayer_regularization_losses
гlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 

Гtrace_0* 

«trace_0* 
* 

ђ0
Ђ1*

ђ0
Ђ1*
* 
ў
»non_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

┤trace_0* 

хtrace_0* 
`Z
VARIABLE_VALUEdense_593/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEdense_593/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
PJ
VARIABLE_VALUEdense_577/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_577/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_582/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_582/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_587/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_587/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_592/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_592/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1bidirectional_92/forward_lstm_92/lstm_cell/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/bidirectional_92/forward_lstm_92/lstm_cell/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE2bidirectional_92/backward_lstm_92/lstm_cell/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE<bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0bidirectional_92/backward_lstm_92/lstm_cell/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
* 
ѓ
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
Ѓ0
ё1*

Ѓ0
ё1*
* 
ъ
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
┤	variables
хtrainable_variables
Хregularization_losses
И__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses*

╗trace_0* 

╝trace_0* 
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
Ё0
є1*

Ё0
є1*
* 
ъ
йnon_trainable_variables
Йlayers
┐metrics
 └layer_regularization_losses
┴layer_metrics
─	variables
┼trainable_variables
кregularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses*

┬trace_0* 

├trace_0* 
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
Є0
ѕ1*

Є0
ѕ1*
* 
ъ
─non_trainable_variables
┼layers
кmetrics
 Кlayer_regularization_losses
╚layer_metrics
н	variables
Нtrainable_variables
оregularization_losses
п__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses*

╔trace_0* 

╩trace_0* 
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
Ѕ0
і1*

Ѕ0
і1*
* 
ъ
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
С	variables
тtrainable_variables
Тregularization_losses
У__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses*

лtrace_0* 

Лtrace_0* 
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

І0
ї1
Ї2*

І0
ї1
Ї2*
* 
Ф
мstates
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
і	variables
Іtrainable_variables
їregularization_losses
ј__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses*
:
пtrace_0
┘trace_1
┌trace_2
█trace_3* 
:
▄trace_0
Пtrace_1
яtrace_2
▀trace_3* 
(
$Я_self_saveable_object_factories* 
ћ
р	variables
Рtrainable_variables
сregularization_losses
С	keras_api
т__call__
+Т&call_and_return_all_conditional_losses
у_random_generator
У
state_size
Іkernel
їrecurrent_kernel
	Їbias
$ж_self_saveable_object_factories*
* 
* 

ј0
Ј1
љ2*

ј0
Ј1
љ2*
* 
Ф
Жstates
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
ћ	variables
Ћtrainable_variables
ќregularization_losses
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses*
:
­trace_0
ыtrace_1
Ыtrace_2
зtrace_3* 
:
Зtrace_0
шtrace_1
Шtrace_2
эtrace_3* 
(
$Э_self_saveable_object_factories* 
ћ
щ	variables
Щtrainable_variables
чregularization_losses
Ч	keras_api
§__call__
+■&call_and_return_all_conditional_losses
 _random_generator
ђ
state_size
јkernel
Јrecurrent_kernel
	љbias
$Ђ_self_saveable_object_factories*
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

Љ0*
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

І0
ї1
Ї2*

І0
ї1
Ї2*
* 
ъ
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
р	variables
Рtrainable_variables
сregularization_losses
т__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses*

Єtrace_0
ѕtrace_1* 

Ѕtrace_0
іtrace_1* 
(
$І_self_saveable_object_factories* 
* 
* 
* 
* 

Џ0*
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

ј0
Ј1
љ2*

ј0
Ј1
љ2*
* 
ъ
їnon_trainable_variables
Їlayers
јmetrics
 Јlayer_regularization_losses
љlayer_metrics
щ	variables
Щtrainable_variables
чregularization_losses
§__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses*

Љtrace_0
њtrace_1* 

Њtrace_0
ћtrace_1* 
(
$Ћ_self_saveable_object_factories* 
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
└
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_576/kerneldense_576/biasdense_593/kerneldense_593/biasdense_577/kerneldense_577/biasdense_582/kerneldense_582/biasdense_587/kerneldense_587/biasdense_592/kerneldense_592/bias1bidirectional_92/forward_lstm_92/lstm_cell/kernel;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel/bidirectional_92/forward_lstm_92/lstm_cell/bias2bidirectional_92/backward_lstm_92/lstm_cell/kernel<bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel0bidirectional_92/backward_lstm_92/lstm_cell/biasConst*
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
GPU 2J 8ѓ **
f%R#
!__inference__traced_save_55814318
╗
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_576/kerneldense_576/biasdense_593/kerneldense_593/biasdense_577/kerneldense_577/biasdense_582/kerneldense_582/biasdense_587/kerneldense_587/biasdense_592/kerneldense_592/bias1bidirectional_92/forward_lstm_92/lstm_cell/kernel;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel/bidirectional_92/forward_lstm_92/lstm_cell/bias2bidirectional_92/backward_lstm_92/lstm_cell/kernel<bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel0bidirectional_92/backward_lstm_92/lstm_cell/bias*
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
GPU 2J 8ѓ *-
f(R&
$__inference__traced_restore_55814381вк2
ъ
d
H__inference_flatten_92_layer_call_and_return_conditional_losses_55812573

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  T
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	ђP
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
╠	
═
while_cond_55812854
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55812854___redundant_placeholder06
2while_while_cond_55812854___redundant_placeholder16
2while_while_cond_55812854___redundant_placeholder26
2while_while_cond_55812854___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
њ┐
ъ
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55811959
inputs_0J
8forward_lstm_92_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_92_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource: 
identityѕб1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpб2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбbackward_lstm_92/whileб0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpб1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбforward_lstm_92/while[
forward_lstm_92/ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤m
#forward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
forward_lstm_92/strided_sliceStridedSliceforward_lstm_92/Shape:output:0,forward_lstm_92/strided_slice/stack:output:0.forward_lstm_92/strided_slice/stack_1:output:0.forward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Б
forward_lstm_92/zeros/packedPack&forward_lstm_92/strided_slice:output:0'forward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
forward_lstm_92/zerosFill%forward_lstm_92/zeros/packed:output:0$forward_lstm_92/zeros/Const:output:0*
T0*'
_output_shapes
:         b
 forward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/zeros_1/packedPack&forward_lstm_92/strided_slice:output:0)forward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    б
forward_lstm_92/zeros_1Fill'forward_lstm_92/zeros_1/packed:output:0&forward_lstm_92/zeros_1/Const:output:0*
T0*'
_output_shapes
:         s
forward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          А
forward_lstm_92/transpose	Transposeinputs_0'forward_lstm_92/transpose/perm:output:0*
T0*=
_output_shapes+
):'                           r
forward_lstm_92/Shape_1Shapeforward_lstm_92/transpose:y:0*
T0*
_output_shapes
::ь¤o
%forward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
forward_lstm_92/strided_slice_1StridedSlice forward_lstm_92/Shape_1:output:0.forward_lstm_92/strided_slice_1/stack:output:00forward_lstm_92/strided_slice_1/stack_1:output:00forward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         С
forward_lstm_92/TensorArrayV2TensorListReserve4forward_lstm_92/TensorArrayV2/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмќ
Eforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        љ
7forward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_92/transpose:y:0Nforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмo
%forward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┬
forward_lstm_92/strided_slice_2StridedSliceforward_lstm_92/transpose:y:0.forward_lstm_92/strided_slice_2/stack:output:00forward_lstm_92/strided_slice_2/stack_1:output:00forward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskе
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0┐
 forward_lstm_92/lstm_cell/MatMulMatMul(forward_lstm_92/strided_slice_2:output:07forward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          г
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0╣
"forward_lstm_92/lstm_cell/MatMul_1MatMulforward_lstm_92/zeros:output:09forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ▓
forward_lstm_92/lstm_cell/addAddV2*forward_lstm_92/lstm_cell/MatMul:product:0,forward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          д
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╗
!forward_lstm_92/lstm_cell/BiasAddBiasAdd!forward_lstm_92/lstm_cell/add:z:08forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          k
)forward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ё
forward_lstm_92/lstm_cell/splitSplit2forward_lstm_92/lstm_cell/split/split_dim:output:0*forward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitѕ
!forward_lstm_92/lstm_cell/SigmoidSigmoid(forward_lstm_92/lstm_cell/split:output:0*
T0*'
_output_shapes
:         і
#forward_lstm_92/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_92/lstm_cell/split:output:1*
T0*'
_output_shapes
:         А
forward_lstm_92/lstm_cell/mulMul'forward_lstm_92/lstm_cell/Sigmoid_1:y:0 forward_lstm_92/zeros_1:output:0*
T0*'
_output_shapes
:         ѓ
forward_lstm_92/lstm_cell/ReluRelu(forward_lstm_92/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Г
forward_lstm_92/lstm_cell/mul_1Mul%forward_lstm_92/lstm_cell/Sigmoid:y:0,forward_lstm_92/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         б
forward_lstm_92/lstm_cell/add_1AddV2!forward_lstm_92/lstm_cell/mul:z:0#forward_lstm_92/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         і
#forward_lstm_92/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_92/lstm_cell/split:output:3*
T0*'
_output_shapes
:         
 forward_lstm_92/lstm_cell/Relu_1Relu#forward_lstm_92/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         ▒
forward_lstm_92/lstm_cell/mul_2Mul'forward_lstm_92/lstm_cell/Sigmoid_2:y:0.forward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ~
-forward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       У
forward_lstm_92/TensorArrayV2_1TensorListReserve6forward_lstm_92/TensorArrayV2_1/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмV
forward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         d
"forward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : └
forward_lstm_92/whileWhile+forward_lstm_92/while/loop_counter:output:01forward_lstm_92/while/maximum_iterations:output:0forward_lstm_92/time:output:0(forward_lstm_92/TensorArrayV2_1:handle:0forward_lstm_92/zeros:output:0 forward_lstm_92/zeros_1:output:0(forward_lstm_92/strided_slice_1:output:0Gforward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_92_lstm_cell_matmul_readvariableop_resource:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#forward_lstm_92_while_body_55811730*/
cond'R%
#forward_lstm_92_while_cond_55811729*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Љ
@forward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ч
2forward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_92/while:output:3Iforward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0x
%forward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         q
'forward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
forward_lstm_92/strided_slice_3StridedSlice;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_92/strided_slice_3/stack:output:00forward_lstm_92/strided_slice_3/stack_1:output:00forward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_masku
 forward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¤
forward_lstm_92/transpose_1	Transpose;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_92/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  k
forward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
backward_lstm_92/ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤n
$backward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
backward_lstm_92/strided_sliceStridedSlicebackward_lstm_92/Shape:output:0-backward_lstm_92/strided_slice/stack:output:0/backward_lstm_92/strided_slice/stack_1:output:0/backward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :д
backward_lstm_92/zeros/packedPack'backward_lstm_92/strided_slice:output:0(backward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
backward_lstm_92/zerosFill&backward_lstm_92/zeros/packed:output:0%backward_lstm_92/zeros/Const:output:0*
T0*'
_output_shapes
:         c
!backward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ф
backward_lstm_92/zeros_1/packedPack'backward_lstm_92/strided_slice:output:0*backward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
backward_lstm_92/zeros_1Fill(backward_lstm_92/zeros_1/packed:output:0'backward_lstm_92/zeros_1/Const:output:0*
T0*'
_output_shapes
:         t
backward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Б
backward_lstm_92/transpose	Transposeinputs_0(backward_lstm_92/transpose/perm:output:0*
T0*=
_output_shapes+
):'                           t
backward_lstm_92/Shape_1Shapebackward_lstm_92/transpose:y:0*
T0*
_output_shapes
::ь¤p
&backward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 backward_lstm_92/strided_slice_1StridedSlice!backward_lstm_92/Shape_1:output:0/backward_lstm_92/strided_slice_1/stack:output:01backward_lstm_92/strided_slice_1/stack_1:output:01backward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         у
backward_lstm_92/TensorArrayV2TensorListReserve5backward_lstm_92/TensorArrayV2/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмi
backward_lstm_92/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ╣
backward_lstm_92/ReverseV2	ReverseV2backward_lstm_92/transpose:y:0(backward_lstm_92/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           Ќ
Fbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ў
8backward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_92/ReverseV2:output:0Obackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмp
&backward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
 backward_lstm_92/strided_slice_2StridedSlicebackward_lstm_92/transpose:y:0/backward_lstm_92/strided_slice_2/stack:output:01backward_lstm_92/strided_slice_2/stack_1:output:01backward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskф
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0┬
!backward_lstm_92/lstm_cell/MatMulMatMul)backward_lstm_92/strided_slice_2:output:08backward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          «
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0╝
#backward_lstm_92/lstm_cell/MatMul_1MatMulbackward_lstm_92/zeros:output:0:backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          х
backward_lstm_92/lstm_cell/addAddV2+backward_lstm_92/lstm_cell/MatMul:product:0-backward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          е
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Й
"backward_lstm_92/lstm_cell/BiasAddBiasAdd"backward_lstm_92/lstm_cell/add:z:09backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          l
*backward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 backward_lstm_92/lstm_cell/splitSplit3backward_lstm_92/lstm_cell/split/split_dim:output:0+backward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitі
"backward_lstm_92/lstm_cell/SigmoidSigmoid)backward_lstm_92/lstm_cell/split:output:0*
T0*'
_output_shapes
:         ї
$backward_lstm_92/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_92/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ц
backward_lstm_92/lstm_cell/mulMul(backward_lstm_92/lstm_cell/Sigmoid_1:y:0!backward_lstm_92/zeros_1:output:0*
T0*'
_output_shapes
:         ё
backward_lstm_92/lstm_cell/ReluRelu)backward_lstm_92/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ░
 backward_lstm_92/lstm_cell/mul_1Mul&backward_lstm_92/lstm_cell/Sigmoid:y:0-backward_lstm_92/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         Ц
 backward_lstm_92/lstm_cell/add_1AddV2"backward_lstm_92/lstm_cell/mul:z:0$backward_lstm_92/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         ї
$backward_lstm_92/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_92/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ђ
!backward_lstm_92/lstm_cell/Relu_1Relu$backward_lstm_92/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         ┤
 backward_lstm_92/lstm_cell/mul_2Mul(backward_lstm_92/lstm_cell/Sigmoid_2:y:0/backward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         
.backward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       в
 backward_lstm_92/TensorArrayV2_1TensorListReserve7backward_lstm_92/TensorArrayV2_1/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмW
backward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         e
#backward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╬
backward_lstm_92/whileWhile,backward_lstm_92/while/loop_counter:output:02backward_lstm_92/while/maximum_iterations:output:0backward_lstm_92/time:output:0)backward_lstm_92/TensorArrayV2_1:handle:0backward_lstm_92/zeros:output:0!backward_lstm_92/zeros_1:output:0)backward_lstm_92/strided_slice_1:output:0Hbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_92_lstm_cell_matmul_readvariableop_resource;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$backward_lstm_92_while_body_55811871*0
cond(R&
$backward_lstm_92_while_cond_55811870*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations њ
Abackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ■
3backward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_92/while:output:3Jbackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0y
&backward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         r
(backward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▄
 backward_lstm_92/strided_slice_3StridedSlice<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_92/strided_slice_3/stack:output:01backward_lstm_92/strided_slice_3/stack_1:output:01backward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskv
!backward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          м
backward_lstm_92/transpose_1	Transpose<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_92/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  l
backward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:љ
	ReverseV2	ReverseV2 backward_lstm_92/transpose_1:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ц
concatConcatV2forward_lstm_92/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  k
IdentityIdentityconcat:output:0^NoOp*
T0*4
_output_shapes"
 :                  ѕ
NoOpNoOp2^backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_92/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_92/while1^forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_92/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 2f
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_92/whilebackward_lstm_92/while2d
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_92/whileforward_lstm_92/while:g c
=
_output_shapes+
):'                           
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
ЌL
▒
#forward_lstm_92_while_body_55811442<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3;
7forward_lstm_92_while_forward_lstm_92_strided_slice_1_0w
sforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_92_while_identity$
 forward_lstm_92_while_identity_1$
 forward_lstm_92_while_identity_2$
 forward_lstm_92_while_identity_3$
 forward_lstm_92_while_identity_4$
 forward_lstm_92_while_identity_59
5forward_lstm_92_while_forward_lstm_92_strided_slice_1u
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpў
Gforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"         
9forward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_92_while_placeholderPforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0Х
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0с
&forward_lstm_92/while/lstm_cell/MatMulMatMul@forward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ║
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0╩
(forward_lstm_92/while/lstm_cell/MatMul_1MatMul#forward_lstm_92_while_placeholder_2?forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ─
#forward_lstm_92/while/lstm_cell/addAddV20forward_lstm_92/while/lstm_cell/MatMul:product:02forward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ┤
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0═
'forward_lstm_92/while/lstm_cell/BiasAddBiasAdd'forward_lstm_92/while/lstm_cell/add:z:0>forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          q
/forward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
%forward_lstm_92/while/lstm_cell/splitSplit8forward_lstm_92/while/lstm_cell/split/split_dim:output:00forward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitћ
'forward_lstm_92/while/lstm_cell/SigmoidSigmoid.forward_lstm_92/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         ќ
)forward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_92/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ░
#forward_lstm_92/while/lstm_cell/mulMul-forward_lstm_92/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_92_while_placeholder_3*
T0*'
_output_shapes
:         ј
$forward_lstm_92/while/lstm_cell/ReluRelu.forward_lstm_92/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ┐
%forward_lstm_92/while/lstm_cell/mul_1Mul+forward_lstm_92/while/lstm_cell/Sigmoid:y:02forward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ┤
%forward_lstm_92/while/lstm_cell/add_1AddV2'forward_lstm_92/while/lstm_cell/mul:z:0)forward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         ќ
)forward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_92/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         І
&forward_lstm_92/while/lstm_cell/Relu_1Relu)forward_lstm_92/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         ├
%forward_lstm_92/while/lstm_cell/mul_2Mul-forward_lstm_92/while/lstm_cell/Sigmoid_2:y:04forward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ѓ
:forward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_92_while_placeholder_1!forward_lstm_92_while_placeholder)forward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм]
forward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ї
forward_lstm_92/while/addAddV2!forward_lstm_92_while_placeholder$forward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/while/add_1AddV28forward_lstm_92_while_forward_lstm_92_while_loop_counter&forward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: Ѕ
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: ф
 forward_lstm_92/while/Identity_1Identity>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ѕ
 forward_lstm_92/while/Identity_2Identityforward_lstm_92/while/add:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Х
 forward_lstm_92/while/Identity_3IdentityJforward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: д
 forward_lstm_92/while/Identity_4Identity)forward_lstm_92/while/lstm_cell/mul_2:z:0^forward_lstm_92/while/NoOp*
T0*'
_output_shapes
:         д
 forward_lstm_92/while/Identity_5Identity)forward_lstm_92/while/lstm_cell/add_1:z:0^forward_lstm_92/while/NoOp*
T0*'
_output_shapes
:         с
forward_lstm_92/while/NoOpNoOp7^forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_92_while_forward_lstm_92_strided_slice_17forward_lstm_92_while_forward_lstm_92_strided_slice_1_0"I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0"M
 forward_lstm_92_while_identity_1)forward_lstm_92/while/Identity_1:output:0"M
 forward_lstm_92_while_identity_2)forward_lstm_92/while/Identity_2:output:0"M
 forward_lstm_92_while_identity_3)forward_lstm_92/while/Identity_3:output:0"M
 forward_lstm_92_while_identity_4)forward_lstm_92/while/Identity_4:output:0"M
 forward_lstm_92_while_identity_5)forward_lstm_92/while/Identity_5:output:0"ё
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"є
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ѓ
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"У
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2p
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_92/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
▄
А
$backward_lstm_92_while_cond_55811870>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3@
<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55811870___redundant_placeholder0X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55811870___redundant_placeholder1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55811870___redundant_placeholder2X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55811870___redundant_placeholder3#
backward_lstm_92_while_identity
д
backward_lstm_92/while/LessLess"backward_lstm_92_while_placeholder<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_92/while/IdentityIdentitybackward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_92/strided_slice_1:

_output_shapes
:
б	
й
3__inference_backward_lstm_92_layer_call_fn_55813401

inputs
unknown: 
	unknown_0: 
	unknown_1: 
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55809803|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_user_specified_name
55813393:($
"
_user_specified_name
55813395:($
"
_user_specified_name
55813397
еJ
љ
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813225

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:Ы
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55813141*
condR
while_cond_55813140*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
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
Б
ђ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809367

inputs

states
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
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
ћ	
Й
2__inference_forward_lstm_92_layer_call_fn_55812774
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55809100|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55812766:($
"
_user_specified_name
55812768:($
"
_user_specified_name
55812770
д
њ
3__inference_bidirectional_92_layer_call_fn_55811349
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55810126|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'                           
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55811335:($
"
_user_specified_name
55811337:($
"
_user_specified_name
55811339:($
"
_user_specified_name
55811341:($
"
_user_specified_name
55811343:($
"
_user_specified_name
55811345
й╣
ю
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55812247

inputsJ
8forward_lstm_92_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_92_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource: 
identityѕб1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpб2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбbackward_lstm_92/whileб0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpб1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбforward_lstm_92/whilej
forward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         m
#forward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
forward_lstm_92/strided_sliceStridedSliceforward_lstm_92/Shape:output:0,forward_lstm_92/strided_slice/stack:output:0.forward_lstm_92/strided_slice/stack_1:output:0.forward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Б
forward_lstm_92/zeros/packedPack&forward_lstm_92/strided_slice:output:0'forward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
forward_lstm_92/zerosFill%forward_lstm_92/zeros/packed:output:0$forward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:b
 forward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/zeros_1/packedPack&forward_lstm_92/strided_slice:output:0)forward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ў
forward_lstm_92/zeros_1Fill'forward_lstm_92/zeros_1/packed:output:0&forward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:s
forward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ё
forward_lstm_92/transpose	Transposeinputs'forward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:l
forward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         o
%forward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
forward_lstm_92/strided_slice_1StridedSlice forward_lstm_92/Shape_1:output:0.forward_lstm_92/strided_slice_1/stack:output:00forward_lstm_92/strided_slice_1/stack_1:output:00forward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         С
forward_lstm_92/TensorArrayV2TensorListReserve4forward_lstm_92/TensorArrayV2/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмќ
Eforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      љ
7forward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_92/transpose:y:0Nforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмo
%forward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
forward_lstm_92/strided_slice_2StridedSliceforward_lstm_92/transpose:y:0.forward_lstm_92/strided_slice_2/stack:output:00forward_lstm_92/strided_slice_2/stack_1:output:00forward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskе
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Х
 forward_lstm_92/lstm_cell/MatMulMatMul(forward_lstm_92/strided_slice_2:output:07forward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: г
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0░
"forward_lstm_92/lstm_cell/MatMul_1MatMulforward_lstm_92/zeros:output:09forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: Е
forward_lstm_92/lstm_cell/addAddV2*forward_lstm_92/lstm_cell/MatMul:product:0,forward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: д
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0▓
!forward_lstm_92/lstm_cell/BiasAddBiasAdd!forward_lstm_92/lstm_cell/add:z:08forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: k
)forward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Я
forward_lstm_92/lstm_cell/splitSplit2forward_lstm_92/lstm_cell/split/split_dim:output:0*forward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split
!forward_lstm_92/lstm_cell/SigmoidSigmoid(forward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:Ђ
#forward_lstm_92/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:ў
forward_lstm_92/lstm_cell/mulMul'forward_lstm_92/lstm_cell/Sigmoid_1:y:0 forward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:y
forward_lstm_92/lstm_cell/ReluRelu(forward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:ц
forward_lstm_92/lstm_cell/mul_1Mul%forward_lstm_92/lstm_cell/Sigmoid:y:0,forward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Ў
forward_lstm_92/lstm_cell/add_1AddV2!forward_lstm_92/lstm_cell/mul:z:0#forward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ђ
#forward_lstm_92/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:v
 forward_lstm_92/lstm_cell/Relu_1Relu#forward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:е
forward_lstm_92/lstm_cell/mul_2Mul'forward_lstm_92/lstm_cell/Sigmoid_2:y:0.forward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:~
-forward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      У
forward_lstm_92/TensorArrayV2_1TensorListReserve6forward_lstm_92/TensorArrayV2_1/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмV
forward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         d
"forward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ю
forward_lstm_92/whileWhile+forward_lstm_92/while/loop_counter:output:01forward_lstm_92/while/maximum_iterations:output:0forward_lstm_92/time:output:0(forward_lstm_92/TensorArrayV2_1:handle:0forward_lstm_92/zeros:output:0 forward_lstm_92/zeros_1:output:0(forward_lstm_92/strided_slice_1:output:0Gforward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_92_lstm_cell_matmul_readvariableop_resource:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
#forward_lstm_92_while_body_55812018*/
cond'R%
#forward_lstm_92_while_cond_55812017*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations Љ
@forward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ж
2forward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_92/while:output:3Iforward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0x
%forward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         q
'forward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
forward_lstm_92/strided_slice_3StridedSlice;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_92/strided_slice_3/stack:output:00forward_lstm_92/strided_slice_3/stack_1:output:00forward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masku
 forward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
forward_lstm_92/transpose_1	Transpose;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:k
forward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
backward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         n
$backward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
backward_lstm_92/strided_sliceStridedSlicebackward_lstm_92/Shape:output:0-backward_lstm_92/strided_slice/stack:output:0/backward_lstm_92/strided_slice/stack_1:output:0/backward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :д
backward_lstm_92/zeros/packedPack'backward_lstm_92/strided_slice:output:0(backward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ќ
backward_lstm_92/zerosFill&backward_lstm_92/zeros/packed:output:0%backward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:c
!backward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ф
backward_lstm_92/zeros_1/packedPack'backward_lstm_92/strided_slice:output:0*backward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
backward_lstm_92/zeros_1Fill(backward_lstm_92/zeros_1/packed:output:0'backward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:t
backward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          є
backward_lstm_92/transpose	Transposeinputs(backward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:m
backward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
&backward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 backward_lstm_92/strided_slice_1StridedSlice!backward_lstm_92/Shape_1:output:0/backward_lstm_92/strided_slice_1/stack:output:01backward_lstm_92/strided_slice_1/stack_1:output:01backward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         у
backward_lstm_92/TensorArrayV2TensorListReserve5backward_lstm_92/TensorArrayV2/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмi
backward_lstm_92/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ъ
backward_lstm_92/ReverseV2	ReverseV2backward_lstm_92/transpose:y:0(backward_lstm_92/ReverseV2/axis:output:0*
T0*"
_output_shapes
:Ќ
Fbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ў
8backward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_92/ReverseV2:output:0Obackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмp
&backward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
 backward_lstm_92/strided_slice_2StridedSlicebackward_lstm_92/transpose:y:0/backward_lstm_92/strided_slice_2/stack:output:01backward_lstm_92/strided_slice_2/stack_1:output:01backward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskф
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0╣
!backward_lstm_92/lstm_cell/MatMulMatMul)backward_lstm_92/strided_slice_2:output:08backward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: «
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0│
#backward_lstm_92/lstm_cell/MatMul_1MatMulbackward_lstm_92/zeros:output:0:backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: г
backward_lstm_92/lstm_cell/addAddV2+backward_lstm_92/lstm_cell/MatMul:product:0-backward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: е
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0х
"backward_lstm_92/lstm_cell/BiasAddBiasAdd"backward_lstm_92/lstm_cell/add:z:09backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: l
*backward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :с
 backward_lstm_92/lstm_cell/splitSplit3backward_lstm_92/lstm_cell/split/split_dim:output:0+backward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitЂ
"backward_lstm_92/lstm_cell/SigmoidSigmoid)backward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:Ѓ
$backward_lstm_92/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:Џ
backward_lstm_92/lstm_cell/mulMul(backward_lstm_92/lstm_cell/Sigmoid_1:y:0!backward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:{
backward_lstm_92/lstm_cell/ReluRelu)backward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:Д
 backward_lstm_92/lstm_cell/mul_1Mul&backward_lstm_92/lstm_cell/Sigmoid:y:0-backward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:ю
 backward_lstm_92/lstm_cell/add_1AddV2"backward_lstm_92/lstm_cell/mul:z:0$backward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ѓ
$backward_lstm_92/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:x
!backward_lstm_92/lstm_cell/Relu_1Relu$backward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:Ф
 backward_lstm_92/lstm_cell/mul_2Mul(backward_lstm_92/lstm_cell/Sigmoid_2:y:0/backward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:
.backward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      в
 backward_lstm_92/TensorArrayV2_1TensorListReserve7backward_lstm_92/TensorArrayV2_1/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмW
backward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         e
#backward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ф
backward_lstm_92/whileWhile,backward_lstm_92/while/loop_counter:output:02backward_lstm_92/while/maximum_iterations:output:0backward_lstm_92/time:output:0)backward_lstm_92/TensorArrayV2_1:handle:0backward_lstm_92/zeros:output:0!backward_lstm_92/zeros_1:output:0)backward_lstm_92/strided_slice_1:output:0Hbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_92_lstm_cell_matmul_readvariableop_resource;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
$backward_lstm_92_while_body_55812159*0
cond(R&
$backward_lstm_92_while_cond_55812158*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations њ
Abackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      В
3backward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_92/while:output:3Jbackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0y
&backward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         r
(backward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
 backward_lstm_92/strided_slice_3StridedSlice<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_92/strided_slice_3/stack:output:01backward_lstm_92/strided_slice_3/stack_1:output:01backward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskv
!backward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          └
backward_lstm_92/transpose_1	Transpose<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:l
backward_lstm_92/runtimeConst"/device:CPU:0*
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
	ReverseV2	ReverseV2 backward_lstm_92/transpose_1:y:0ReverseV2/axis:output:0*
T0*"
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Њ
concatConcatV2forward_lstm_92/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*"
_output_shapes
:Y
IdentityIdentityconcat:output:0^NoOp*
T0*"
_output_shapes
:ѕ
NoOpNoOp2^backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_92/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_92/while1^forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_92/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2f
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_92/whilebackward_lstm_92/while2d
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_92/whileforward_lstm_92/while:J F
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
і

g
H__inference_dropout_92_layer_call_and_return_conditional_losses_55812557

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?_
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
:b
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         Є
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
 *═╠L>А
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ј
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
д
г
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_cond_55808267~
ztopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_loop_counterЁ
ђtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_maximum_iterationsF
Btopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholderH
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder_1H
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder_2H
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder_3ђ
|topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_less_topk_bilstm_moe_bidirectional_92_forward_lstm_92_strided_slice_1Ў
ћtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_cond_55808267___redundant_placeholder0Ў
ћtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_cond_55808267___redundant_placeholder1Ў
ћtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_cond_55808267___redundant_placeholder2Ў
ћtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_cond_55808267___redundant_placeholder3C
?topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity
д
;topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/LessLessBtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder|topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_less_topk_bilstm_moe_bidirectional_92_forward_lstm_92_strided_slice_1*
T0*
_output_shapes
: Г
?topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/IdentityIdentity?topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "І
?topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identityHtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::{ w

_output_shapes
: 
]
_user_specified_nameECtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/loop_counter:Ђ}

_output_shapes
: 
c
_user_specified_nameKItopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/maximum_iterations:
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
_user_specified_nameB@topk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1:

_output_shapes
:
Б
ђ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55808872

inputs

states
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
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
С$
о
while_body_55809381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_55809405_0: ,
while_lstm_cell_55809407_0: (
while_lstm_cell_55809409_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_55809405: *
while_lstm_cell_55809407: &
while_lstm_cell_55809409: ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55809405_0while_lstm_cell_55809407_0while_lstm_cell_55809409_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809367┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_55809405while_lstm_cell_55809405_0"6
while_lstm_cell_55809407while_lstm_cell_55809407_0"6
while_lstm_cell_55809409while_lstm_cell_55809409_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
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
:         :-)
'
_output_shapes
:         :GC
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
55809405:(	$
"
_user_specified_name
55809407:(
$
"
_user_specified_name
55809409
ѕ
Ў
,__inference_dense_587_layer_call_fn_55812681

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_587_layer_call_and_return_conditional_losses_55808691s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
55812675:($
"
_user_specified_name
55812677
Д	
Я
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808622
dense_582_input$
dense_582_55808616: 
dense_582_55808618:
identityѕб!dense_582/StatefulPartitionedCallЄ
!dense_582/StatefulPartitionedCallStatefulPartitionedCalldense_582_inputdense_582_55808616dense_582_55808618*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_582_layer_call_and_return_conditional_losses_55808615}
IdentityIdentity*dense_582/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         F
NoOpNoOp"^dense_582/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 2F
!dense_582/StatefulPartitionedCall!dense_582/StatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_582_input:($
"
_user_specified_name
55808616:($
"
_user_specified_name
55808618
ЉL
Љ
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55809803

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: є
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        т
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:Ы
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55809719*
condR
while_cond_55809718*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
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
ыH
ѕ	
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810617
input_3$
dense_576_55810222: 
dense_576_55810224:)
sequential_392_55810252:%
sequential_392_55810254:)
sequential_397_55810257:%
sequential_397_55810259:)
sequential_402_55810262:%
sequential_402_55810264:)
sequential_407_55810267:%
sequential_407_55810269:+
bidirectional_92_55810568: +
bidirectional_92_55810570: '
bidirectional_92_55810572: +
bidirectional_92_55810574: +
bidirectional_92_55810576: '
bidirectional_92_55810578: %
dense_593_55810611:	ђ 
dense_593_55810613:
identityѕб(bidirectional_92/StatefulPartitionedCallб!dense_576/StatefulPartitionedCallб!dense_593/StatefulPartitionedCallб"dropout_92/StatefulPartitionedCallб&sequential_392/StatefulPartitionedCallб&sequential_397/StatefulPartitionedCallб&sequential_402/StatefulPartitionedCallб&sequential_407/StatefulPartitionedCallШ
!dense_576/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_576_55810222dense_576_55810224*
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
GPU 2J 8ѓ *P
fKRI
G__inference_dense_576_layer_call_and_return_conditional_losses_55810221Z
tf.math.top_k_2/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :ф
tf.math.top_k_2/TopKV2TopKV2*dense_576/StatefulPartitionedCall:output:0!tf.math.top_k_2/TopKV2/k:output:0*
T0*0
_output_shapes
::b
tf.one_hot_2/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?c
tf.one_hot_2/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    \
tf.one_hot_2/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :ы
tf.one_hot_2/one_hotOneHot tf.math.top_k_2/TopKV2:indices:0#tf.one_hot_2/one_hot/depth:output:0&tf.one_hot_2/one_hot/on_value:output:0'tf.one_hot_2/one_hot/off_value:output:0*
T0*
TI0*&
_output_shapes
:­
lambda_96/PartitionedCallPartitionedCalltf.math.top_k_2/TopKV2:values:0tf.one_hot_2/one_hot:output:0*
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_96_layer_call_and_return_conditional_losses_55810239р
lambda_97/PartitionedCallPartitionedCallinput_3"lambda_96/PartitionedCall:output:0*
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_97_layer_call_and_return_conditional_losses_55810246ц
tf.unstack_2/unstackUnpack"lambda_97/PartitionedCall:output:0*
T0*L
_output_shapes:
8::::*	
numа
&sequential_392/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_2/unstack:output:0sequential_392_55810252sequential_392_55810254*
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
GPU 2J 8ѓ *U
fPRN
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808546а
&sequential_397/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_2/unstack:output:1sequential_397_55810257sequential_397_55810259*
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
GPU 2J 8ѓ *U
fPRN
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808622а
&sequential_402/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_2/unstack:output:2sequential_402_55810262sequential_402_55810264*
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
GPU 2J 8ѓ *U
fPRN
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808698а
&sequential_407/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_2/unstack:output:3sequential_407_55810267sequential_407_55810269*
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
GPU 2J 8ѓ *U
fPRN
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808774Б
tf.stack_93/stackPack/sequential_392/StatefulPartitionedCall:output:0/sequential_397/StatefulPartitionedCall:output:0/sequential_402/StatefulPartitionedCall:output:0/sequential_407/StatefulPartitionedCall:output:0*
N*
T0*&
_output_shapes
:*

axis­
lambda_98/PartitionedCallPartitionedCalltf.stack_93/stack:output:0"lambda_96/PartitionedCall:output:0*
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_98_layer_call_and_return_conditional_losses_55810278А
(bidirectional_92/StatefulPartitionedCallStatefulPartitionedCall"lambda_98/PartitionedCall:output:0bidirectional_92_55810568bidirectional_92_55810570bidirectional_92_55810572bidirectional_92_55810574bidirectional_92_55810576bidirectional_92_55810578*
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
GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55810567З
"dropout_92/StatefulPartitionedCallStatefulPartitionedCall1bidirectional_92/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_92_layer_call_and_return_conditional_losses_55810592█
flatten_92/PartitionedCallPartitionedCall+dropout_92/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_flatten_92_layer_call_and_return_conditional_losses_55810599ј
!dense_593/StatefulPartitionedCallStatefulPartitionedCall#flatten_92/PartitionedCall:output:0dense_593_55810611dense_593_55810613*
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
GPU 2J 8ѓ *P
fKRI
G__inference_dense_593_layer_call_and_return_conditional_losses_55810610p
IdentityIdentity*dense_593/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:я
NoOpNoOp)^bidirectional_92/StatefulPartitionedCall"^dense_576/StatefulPartitionedCall"^dense_593/StatefulPartitionedCall#^dropout_92/StatefulPartitionedCall'^sequential_392/StatefulPartitionedCall'^sequential_397/StatefulPartitionedCall'^sequential_402/StatefulPartitionedCall'^sequential_407/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 2T
(bidirectional_92/StatefulPartitionedCall(bidirectional_92/StatefulPartitionedCall2F
!dense_576/StatefulPartitionedCall!dense_576/StatefulPartitionedCall2F
!dense_593/StatefulPartitionedCall!dense_593/StatefulPartitionedCall2H
"dropout_92/StatefulPartitionedCall"dropout_92/StatefulPartitionedCall2P
&sequential_392/StatefulPartitionedCall&sequential_392/StatefulPartitionedCall2P
&sequential_397/StatefulPartitionedCall&sequential_397/StatefulPartitionedCall2P
&sequential_402/StatefulPartitionedCall&sequential_402/StatefulPartitionedCall2P
&sequential_407/StatefulPartitionedCall&sequential_407/StatefulPartitionedCall:K G
"
_output_shapes
:
!
_user_specified_name	input_3:($
"
_user_specified_name
55810222:($
"
_user_specified_name
55810224:($
"
_user_specified_name
55810252:($
"
_user_specified_name
55810254:($
"
_user_specified_name
55810257:($
"
_user_specified_name
55810259:($
"
_user_specified_name
55810262:($
"
_user_specified_name
55810264:(	$
"
_user_specified_name
55810267:(
$
"
_user_specified_name
55810269:($
"
_user_specified_name
55810568:($
"
_user_specified_name
55810570:($
"
_user_specified_name
55810572:($
"
_user_specified_name
55810574:($
"
_user_specified_name
55810576:($
"
_user_specified_name
55810578:($
"
_user_specified_name
55810611:($
"
_user_specified_name
55810613
ѕ
Ў
,__inference_dense_577_layer_call_fn_55812601

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_577_layer_call_and_return_conditional_losses_55808539s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
55812595:($
"
_user_specified_name
55812597
ХJ
▒
#forward_lstm_92_while_body_55810338<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3;
7forward_lstm_92_while_forward_lstm_92_strided_slice_1_0w
sforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_92_while_identity$
 forward_lstm_92_while_identity_1$
 forward_lstm_92_while_identity_2$
 forward_lstm_92_while_identity_3$
 forward_lstm_92_while_identity_4$
 forward_lstm_92_while_identity_59
5forward_lstm_92_while_forward_lstm_92_strided_slice_1u
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpў
Gforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ь
9forward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_92_while_placeholderPforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0Х
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0┌
&forward_lstm_92/while/lstm_cell/MatMulMatMul@forward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ║
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0┴
(forward_lstm_92/while/lstm_cell/MatMul_1MatMul#forward_lstm_92_while_placeholder_2?forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: ╗
#forward_lstm_92/while/lstm_cell/addAddV20forward_lstm_92/while/lstm_cell/MatMul:product:02forward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: ┤
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0─
'forward_lstm_92/while/lstm_cell/BiasAddBiasAdd'forward_lstm_92/while/lstm_cell/add:z:0>forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
/forward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
%forward_lstm_92/while/lstm_cell/splitSplit8forward_lstm_92/while/lstm_cell/split/split_dim:output:00forward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitІ
'forward_lstm_92/while/lstm_cell/SigmoidSigmoid.forward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:Ї
)forward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:Д
#forward_lstm_92/while/lstm_cell/mulMul-forward_lstm_92/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:Ё
$forward_lstm_92/while/lstm_cell/ReluRelu.forward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:Х
%forward_lstm_92/while/lstm_cell/mul_1Mul+forward_lstm_92/while/lstm_cell/Sigmoid:y:02forward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Ф
%forward_lstm_92/while/lstm_cell/add_1AddV2'forward_lstm_92/while/lstm_cell/mul:z:0)forward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ї
)forward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:ѓ
&forward_lstm_92/while/lstm_cell/Relu_1Relu)forward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:║
%forward_lstm_92/while/lstm_cell/mul_2Mul-forward_lstm_92/while/lstm_cell/Sigmoid_2:y:04forward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:ѓ
:forward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_92_while_placeholder_1!forward_lstm_92_while_placeholder)forward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм]
forward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ї
forward_lstm_92/while/addAddV2!forward_lstm_92_while_placeholder$forward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/while/add_1AddV28forward_lstm_92_while_forward_lstm_92_while_loop_counter&forward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: Ѕ
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: ф
 forward_lstm_92/while/Identity_1Identity>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ѕ
 forward_lstm_92/while/Identity_2Identityforward_lstm_92/while/add:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Х
 forward_lstm_92/while/Identity_3IdentityJforward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ю
 forward_lstm_92/while/Identity_4Identity)forward_lstm_92/while/lstm_cell/mul_2:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes

:Ю
 forward_lstm_92/while/Identity_5Identity)forward_lstm_92/while/lstm_cell/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes

:с
forward_lstm_92/while/NoOpNoOp7^forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_92_while_forward_lstm_92_strided_slice_17forward_lstm_92_while_forward_lstm_92_strided_slice_1_0"I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0"M
 forward_lstm_92_while_identity_1)forward_lstm_92/while/Identity_1:output:0"M
 forward_lstm_92_while_identity_2)forward_lstm_92/while/Identity_2:output:0"M
 forward_lstm_92_while_identity_3)forward_lstm_92/while/Identity_3:output:0"M
 forward_lstm_92_while_identity_4)forward_lstm_92/while/Identity_4:output:0"M
 forward_lstm_92_while_identity_5)forward_lstm_92/while/Identity_5:output:0"ё
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"є
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ѓ
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"У
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2p
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:
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
_user_specified_name!forward_lstm_92/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
П8
є
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55809100

inputs$
lstm_cell_55809018: $
lstm_cell_55809020:  
lstm_cell_55809022: 
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЬ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55809018lstm_cell_55809020lstm_cell_55809022*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809017n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55809018lstm_cell_55809020lstm_cell_55809022*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55809031*
condR
while_cond_55809030*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:($
"
_user_specified_name
55809018:($
"
_user_specified_name
55809020:($
"
_user_specified_name
55809022
С
■
G__inference_dense_587_layer_call_and_return_conditional_losses_55812712

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
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
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : ю
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
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
▀8
▒
while_body_55812998
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
▀8
▒
while_body_55813473
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
┼G
с
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810992
input_3$
dense_576_55810620: 
dense_576_55810622:)
sequential_392_55810650:%
sequential_392_55810652:)
sequential_397_55810655:%
sequential_397_55810657:)
sequential_402_55810660:%
sequential_402_55810662:)
sequential_407_55810665:%
sequential_407_55810667:+
bidirectional_92_55810966: +
bidirectional_92_55810968: '
bidirectional_92_55810970: +
bidirectional_92_55810972: +
bidirectional_92_55810974: '
bidirectional_92_55810976: %
dense_593_55810986:	ђ 
dense_593_55810988:
identityѕб(bidirectional_92/StatefulPartitionedCallб!dense_576/StatefulPartitionedCallб!dense_593/StatefulPartitionedCallб&sequential_392/StatefulPartitionedCallб&sequential_397/StatefulPartitionedCallб&sequential_402/StatefulPartitionedCallб&sequential_407/StatefulPartitionedCallШ
!dense_576/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_576_55810620dense_576_55810622*
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
GPU 2J 8ѓ *P
fKRI
G__inference_dense_576_layer_call_and_return_conditional_losses_55810221Z
tf.math.top_k_2/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :ф
tf.math.top_k_2/TopKV2TopKV2*dense_576/StatefulPartitionedCall:output:0!tf.math.top_k_2/TopKV2/k:output:0*
T0*0
_output_shapes
::b
tf.one_hot_2/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?c
tf.one_hot_2/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    \
tf.one_hot_2/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :ы
tf.one_hot_2/one_hotOneHot tf.math.top_k_2/TopKV2:indices:0#tf.one_hot_2/one_hot/depth:output:0&tf.one_hot_2/one_hot/on_value:output:0'tf.one_hot_2/one_hot/off_value:output:0*
T0*
TI0*&
_output_shapes
:­
lambda_96/PartitionedCallPartitionedCalltf.math.top_k_2/TopKV2:values:0tf.one_hot_2/one_hot:output:0*
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_96_layer_call_and_return_conditional_losses_55810637р
lambda_97/PartitionedCallPartitionedCallinput_3"lambda_96/PartitionedCall:output:0*
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_97_layer_call_and_return_conditional_losses_55810644ц
tf.unstack_2/unstackUnpack"lambda_97/PartitionedCall:output:0*
T0*L
_output_shapes:
8::::*	
numа
&sequential_392/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_2/unstack:output:0sequential_392_55810650sequential_392_55810652*
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
GPU 2J 8ѓ *U
fPRN
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808555а
&sequential_397/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_2/unstack:output:1sequential_397_55810655sequential_397_55810657*
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
GPU 2J 8ѓ *U
fPRN
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808631а
&sequential_402/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_2/unstack:output:2sequential_402_55810660sequential_402_55810662*
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
GPU 2J 8ѓ *U
fPRN
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808707а
&sequential_407/StatefulPartitionedCallStatefulPartitionedCalltf.unstack_2/unstack:output:3sequential_407_55810665sequential_407_55810667*
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
GPU 2J 8ѓ *U
fPRN
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808783Б
tf.stack_93/stackPack/sequential_392/StatefulPartitionedCall:output:0/sequential_397/StatefulPartitionedCall:output:0/sequential_402/StatefulPartitionedCall:output:0/sequential_407/StatefulPartitionedCall:output:0*
N*
T0*&
_output_shapes
:*

axis­
lambda_98/PartitionedCallPartitionedCalltf.stack_93/stack:output:0"lambda_96/PartitionedCall:output:0*
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_98_layer_call_and_return_conditional_losses_55810676А
(bidirectional_92/StatefulPartitionedCallStatefulPartitionedCall"lambda_98/PartitionedCall:output:0bidirectional_92_55810966bidirectional_92_55810968bidirectional_92_55810970bidirectional_92_55810972bidirectional_92_55810974bidirectional_92_55810976*
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
GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55810965С
dropout_92/PartitionedCallPartitionedCall1bidirectional_92/StatefulPartitionedCall:output:0*
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
GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_92_layer_call_and_return_conditional_losses_55810983М
flatten_92/PartitionedCallPartitionedCall#dropout_92/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_flatten_92_layer_call_and_return_conditional_losses_55810599ј
!dense_593/StatefulPartitionedCallStatefulPartitionedCall#flatten_92/PartitionedCall:output:0dense_593_55810986dense_593_55810988*
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
GPU 2J 8ѓ *P
fKRI
G__inference_dense_593_layer_call_and_return_conditional_losses_55810610p
IdentityIdentity*dense_593/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:╣
NoOpNoOp)^bidirectional_92/StatefulPartitionedCall"^dense_576/StatefulPartitionedCall"^dense_593/StatefulPartitionedCall'^sequential_392/StatefulPartitionedCall'^sequential_397/StatefulPartitionedCall'^sequential_402/StatefulPartitionedCall'^sequential_407/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 2T
(bidirectional_92/StatefulPartitionedCall(bidirectional_92/StatefulPartitionedCall2F
!dense_576/StatefulPartitionedCall!dense_576/StatefulPartitionedCall2F
!dense_593/StatefulPartitionedCall!dense_593/StatefulPartitionedCall2P
&sequential_392/StatefulPartitionedCall&sequential_392/StatefulPartitionedCall2P
&sequential_397/StatefulPartitionedCall&sequential_397/StatefulPartitionedCall2P
&sequential_402/StatefulPartitionedCall&sequential_402/StatefulPartitionedCall2P
&sequential_407/StatefulPartitionedCall&sequential_407/StatefulPartitionedCall:K G
"
_output_shapes
:
!
_user_specified_name	input_3:($
"
_user_specified_name
55810620:($
"
_user_specified_name
55810622:($
"
_user_specified_name
55810650:($
"
_user_specified_name
55810652:($
"
_user_specified_name
55810655:($
"
_user_specified_name
55810657:($
"
_user_specified_name
55810660:($
"
_user_specified_name
55810662:(	$
"
_user_specified_name
55810665:(
$
"
_user_specified_name
55810667:($
"
_user_specified_name
55810966:($
"
_user_specified_name
55810968:($
"
_user_specified_name
55810970:($
"
_user_specified_name
55810972:($
"
_user_specified_name
55810974:($
"
_user_specified_name
55810976:($
"
_user_specified_name
55810986:($
"
_user_specified_name
55810988
Ь
s
G__inference_lambda_96_layer_call_and_return_conditional_losses_55811261
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
С
■
G__inference_dense_592_layer_call_and_return_conditional_losses_55808767

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
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
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : ю
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
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
њ
X
,__inference_lambda_97_layer_call_fn_55811279
inputs_0
inputs_1
identityЙ
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_97_layer_call_and_return_conditional_losses_55810644_
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
╠	
═
while_cond_55813907
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55813907___redundant_placeholder06
2while_while_cond_55813907___redundant_placeholder16
2while_while_cond_55813907___redundant_placeholder26
2while_while_cond_55813907___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
Д	
Я
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808783
dense_592_input$
dense_592_55808777: 
dense_592_55808779:
identityѕб!dense_592/StatefulPartitionedCallЄ
!dense_592/StatefulPartitionedCallStatefulPartitionedCalldense_592_inputdense_592_55808777dense_592_55808779*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_592_layer_call_and_return_conditional_losses_55808767}
IdentityIdentity*dense_592/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         F
NoOpNoOp"^dense_592/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 2F
!dense_592/StatefulPartitionedCall!dense_592/StatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_592_input:($
"
_user_specified_name
55808777:($
"
_user_specified_name
55808779
Ѕљ
Ф
#__inference__wrapped_model_55808506
input_3M
;topk_bilstm_moe_dense_576_tensordot_readvariableop_resource:G
9topk_bilstm_moe_dense_576_biasadd_readvariableop_resource:\
Jtopk_bilstm_moe_sequential_392_dense_577_tensordot_readvariableop_resource:V
Htopk_bilstm_moe_sequential_392_dense_577_biasadd_readvariableop_resource:\
Jtopk_bilstm_moe_sequential_397_dense_582_tensordot_readvariableop_resource:V
Htopk_bilstm_moe_sequential_397_dense_582_biasadd_readvariableop_resource:\
Jtopk_bilstm_moe_sequential_402_dense_587_tensordot_readvariableop_resource:V
Htopk_bilstm_moe_sequential_402_dense_587_biasadd_readvariableop_resource:\
Jtopk_bilstm_moe_sequential_407_dense_592_tensordot_readvariableop_resource:V
Htopk_bilstm_moe_sequential_407_dense_592_biasadd_readvariableop_resource:k
Ytopk_bilstm_moe_bidirectional_92_forward_lstm_92_lstm_cell_matmul_readvariableop_resource: m
[topk_bilstm_moe_bidirectional_92_forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: h
Ztopk_bilstm_moe_bidirectional_92_forward_lstm_92_lstm_cell_biasadd_readvariableop_resource: l
Ztopk_bilstm_moe_bidirectional_92_backward_lstm_92_lstm_cell_matmul_readvariableop_resource: n
\topk_bilstm_moe_bidirectional_92_backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: i
[topk_bilstm_moe_bidirectional_92_backward_lstm_92_lstm_cell_biasadd_readvariableop_resource: K
8topk_bilstm_moe_dense_593_matmul_readvariableop_resource:	ђG
9topk_bilstm_moe_dense_593_biasadd_readvariableop_resource:
identityѕбRtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpбQtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul/ReadVariableOpбStopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpб7topk_bilstm_moe/bidirectional_92/backward_lstm_92/whileбQtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpбPtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpбRtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpб6topk_bilstm_moe/bidirectional_92/forward_lstm_92/whileб0topk_bilstm_moe/dense_576/BiasAdd/ReadVariableOpб2topk_bilstm_moe/dense_576/Tensordot/ReadVariableOpб0topk_bilstm_moe/dense_593/BiasAdd/ReadVariableOpб/topk_bilstm_moe/dense_593/MatMul/ReadVariableOpб?topk_bilstm_moe/sequential_392/dense_577/BiasAdd/ReadVariableOpбAtopk_bilstm_moe/sequential_392/dense_577/Tensordot/ReadVariableOpб?topk_bilstm_moe/sequential_397/dense_582/BiasAdd/ReadVariableOpбAtopk_bilstm_moe/sequential_397/dense_582/Tensordot/ReadVariableOpб?topk_bilstm_moe/sequential_402/dense_587/BiasAdd/ReadVariableOpбAtopk_bilstm_moe/sequential_402/dense_587/Tensordot/ReadVariableOpб?topk_bilstm_moe/sequential_407/dense_592/BiasAdd/ReadVariableOpбAtopk_bilstm_moe/sequential_407/dense_592/Tensordot/ReadVariableOp«
2topk_bilstm_moe/dense_576/Tensordot/ReadVariableOpReadVariableOp;topk_bilstm_moe_dense_576_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0ѓ
1topk_bilstm_moe/dense_576/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ђ     Ц
+topk_bilstm_moe/dense_576/Tensordot/ReshapeReshapeinput_3:topk_bilstm_moe/dense_576/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	ђл
*topk_bilstm_moe/dense_576/Tensordot/MatMulMatMul4topk_bilstm_moe/dense_576/Tensordot/Reshape:output:0:topk_bilstm_moe/dense_576/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђ~
)topk_bilstm_moe/dense_576/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ┼
#topk_bilstm_moe/dense_576/TensordotReshape4topk_bilstm_moe/dense_576/Tensordot/MatMul:product:02topk_bilstm_moe/dense_576/Tensordot/shape:output:0*
T0*"
_output_shapes
:д
0topk_bilstm_moe/dense_576/BiasAdd/ReadVariableOpReadVariableOp9topk_bilstm_moe_dense_576_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┴
!topk_bilstm_moe/dense_576/BiasAddBiasAdd,topk_bilstm_moe/dense_576/Tensordot:output:08topk_bilstm_moe/dense_576/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ё
!topk_bilstm_moe/dense_576/SoftmaxSoftmax*topk_bilstm_moe/dense_576/BiasAdd:output:0*
T0*"
_output_shapes
:j
(topk_bilstm_moe/tf.math.top_k_2/TopKV2/kConst*
_output_shapes
: *
dtype0*
value	B :╦
&topk_bilstm_moe/tf.math.top_k_2/TopKV2TopKV2+topk_bilstm_moe/dense_576/Softmax:softmax:01topk_bilstm_moe/tf.math.top_k_2/TopKV2/k:output:0*
T0*0
_output_shapes
::r
-topk_bilstm_moe/tf.one_hot_2/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?s
.topk_bilstm_moe/tf.one_hot_2/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    l
*topk_bilstm_moe/tf.one_hot_2/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :┴
$topk_bilstm_moe/tf.one_hot_2/one_hotOneHot0topk_bilstm_moe/tf.math.top_k_2/TopKV2:indices:03topk_bilstm_moe/tf.one_hot_2/one_hot/depth:output:06topk_bilstm_moe/tf.one_hot_2/one_hot/on_value:output:07topk_bilstm_moe/tf.one_hot_2/one_hot/off_value:output:0*
T0*
TI0*&
_output_shapes
:С
'topk_bilstm_moe/lambda_96/einsum/EinsumEinsum/topk_bilstm_moe/tf.math.top_k_2/TopKV2:values:0-topk_bilstm_moe/tf.one_hot_2/one_hot:output:0*
N*
T0*"
_output_shapes
:*
equationabc,abcd->abd├
'topk_bilstm_moe/lambda_97/einsum/EinsumEinsuminput_30topk_bilstm_moe/lambda_96/einsum/Einsum:output:0*
N*
T0*&
_output_shapes
:*
equationabc,abd->dabc┬
$topk_bilstm_moe/tf.unstack_2/unstackUnpack0topk_bilstm_moe/lambda_97/einsum/Einsum:output:0*
T0*L
_output_shapes:
8::::*	
num╠
Atopk_bilstm_moe/sequential_392/dense_577/Tensordot/ReadVariableOpReadVariableOpJtopk_bilstm_moe_sequential_392_dense_577_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0Љ
@topk_bilstm_moe/sequential_392/dense_577/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ђ     ж
:topk_bilstm_moe/sequential_392/dense_577/Tensordot/ReshapeReshape-topk_bilstm_moe/tf.unstack_2/unstack:output:0Itopk_bilstm_moe/sequential_392/dense_577/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	ђ§
9topk_bilstm_moe/sequential_392/dense_577/Tensordot/MatMulMatMulCtopk_bilstm_moe/sequential_392/dense_577/Tensordot/Reshape:output:0Itopk_bilstm_moe/sequential_392/dense_577/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђЇ
8topk_bilstm_moe/sequential_392/dense_577/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ы
2topk_bilstm_moe/sequential_392/dense_577/TensordotReshapeCtopk_bilstm_moe/sequential_392/dense_577/Tensordot/MatMul:product:0Atopk_bilstm_moe/sequential_392/dense_577/Tensordot/shape:output:0*
T0*"
_output_shapes
:─
?topk_bilstm_moe/sequential_392/dense_577/BiasAdd/ReadVariableOpReadVariableOpHtopk_bilstm_moe_sequential_392_dense_577_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
0topk_bilstm_moe/sequential_392/dense_577/BiasAddBiasAdd;topk_bilstm_moe/sequential_392/dense_577/Tensordot:output:0Gtopk_bilstm_moe/sequential_392/dense_577/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ю
-topk_bilstm_moe/sequential_392/dense_577/ReluRelu9topk_bilstm_moe/sequential_392/dense_577/BiasAdd:output:0*
T0*"
_output_shapes
:╠
Atopk_bilstm_moe/sequential_397/dense_582/Tensordot/ReadVariableOpReadVariableOpJtopk_bilstm_moe_sequential_397_dense_582_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0Љ
@topk_bilstm_moe/sequential_397/dense_582/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ђ     ж
:topk_bilstm_moe/sequential_397/dense_582/Tensordot/ReshapeReshape-topk_bilstm_moe/tf.unstack_2/unstack:output:1Itopk_bilstm_moe/sequential_397/dense_582/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	ђ§
9topk_bilstm_moe/sequential_397/dense_582/Tensordot/MatMulMatMulCtopk_bilstm_moe/sequential_397/dense_582/Tensordot/Reshape:output:0Itopk_bilstm_moe/sequential_397/dense_582/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђЇ
8topk_bilstm_moe/sequential_397/dense_582/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ы
2topk_bilstm_moe/sequential_397/dense_582/TensordotReshapeCtopk_bilstm_moe/sequential_397/dense_582/Tensordot/MatMul:product:0Atopk_bilstm_moe/sequential_397/dense_582/Tensordot/shape:output:0*
T0*"
_output_shapes
:─
?topk_bilstm_moe/sequential_397/dense_582/BiasAdd/ReadVariableOpReadVariableOpHtopk_bilstm_moe_sequential_397_dense_582_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
0topk_bilstm_moe/sequential_397/dense_582/BiasAddBiasAdd;topk_bilstm_moe/sequential_397/dense_582/Tensordot:output:0Gtopk_bilstm_moe/sequential_397/dense_582/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ю
-topk_bilstm_moe/sequential_397/dense_582/ReluRelu9topk_bilstm_moe/sequential_397/dense_582/BiasAdd:output:0*
T0*"
_output_shapes
:╠
Atopk_bilstm_moe/sequential_402/dense_587/Tensordot/ReadVariableOpReadVariableOpJtopk_bilstm_moe_sequential_402_dense_587_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0Љ
@topk_bilstm_moe/sequential_402/dense_587/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ђ     ж
:topk_bilstm_moe/sequential_402/dense_587/Tensordot/ReshapeReshape-topk_bilstm_moe/tf.unstack_2/unstack:output:2Itopk_bilstm_moe/sequential_402/dense_587/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	ђ§
9topk_bilstm_moe/sequential_402/dense_587/Tensordot/MatMulMatMulCtopk_bilstm_moe/sequential_402/dense_587/Tensordot/Reshape:output:0Itopk_bilstm_moe/sequential_402/dense_587/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђЇ
8topk_bilstm_moe/sequential_402/dense_587/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ы
2topk_bilstm_moe/sequential_402/dense_587/TensordotReshapeCtopk_bilstm_moe/sequential_402/dense_587/Tensordot/MatMul:product:0Atopk_bilstm_moe/sequential_402/dense_587/Tensordot/shape:output:0*
T0*"
_output_shapes
:─
?topk_bilstm_moe/sequential_402/dense_587/BiasAdd/ReadVariableOpReadVariableOpHtopk_bilstm_moe_sequential_402_dense_587_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
0topk_bilstm_moe/sequential_402/dense_587/BiasAddBiasAdd;topk_bilstm_moe/sequential_402/dense_587/Tensordot:output:0Gtopk_bilstm_moe/sequential_402/dense_587/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ю
-topk_bilstm_moe/sequential_402/dense_587/ReluRelu9topk_bilstm_moe/sequential_402/dense_587/BiasAdd:output:0*
T0*"
_output_shapes
:╠
Atopk_bilstm_moe/sequential_407/dense_592/Tensordot/ReadVariableOpReadVariableOpJtopk_bilstm_moe_sequential_407_dense_592_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0Љ
@topk_bilstm_moe/sequential_407/dense_592/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ђ     ж
:topk_bilstm_moe/sequential_407/dense_592/Tensordot/ReshapeReshape-topk_bilstm_moe/tf.unstack_2/unstack:output:3Itopk_bilstm_moe/sequential_407/dense_592/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	ђ§
9topk_bilstm_moe/sequential_407/dense_592/Tensordot/MatMulMatMulCtopk_bilstm_moe/sequential_407/dense_592/Tensordot/Reshape:output:0Itopk_bilstm_moe/sequential_407/dense_592/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђЇ
8topk_bilstm_moe/sequential_407/dense_592/Tensordot/shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ы
2topk_bilstm_moe/sequential_407/dense_592/TensordotReshapeCtopk_bilstm_moe/sequential_407/dense_592/Tensordot/MatMul:product:0Atopk_bilstm_moe/sequential_407/dense_592/Tensordot/shape:output:0*
T0*"
_output_shapes
:─
?topk_bilstm_moe/sequential_407/dense_592/BiasAdd/ReadVariableOpReadVariableOpHtopk_bilstm_moe_sequential_407_dense_592_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ь
0topk_bilstm_moe/sequential_407/dense_592/BiasAddBiasAdd;topk_bilstm_moe/sequential_407/dense_592/Tensordot:output:0Gtopk_bilstm_moe/sequential_407/dense_592/BiasAdd/ReadVariableOp:value:0*
T0*"
_output_shapes
:Ю
-topk_bilstm_moe/sequential_407/dense_592/ReluRelu9topk_bilstm_moe/sequential_407/dense_592/BiasAdd:output:0*
T0*"
_output_shapes
:с
!topk_bilstm_moe/tf.stack_93/stackPack;topk_bilstm_moe/sequential_392/dense_577/Relu:activations:0;topk_bilstm_moe/sequential_397/dense_582/Relu:activations:0;topk_bilstm_moe/sequential_402/dense_587/Relu:activations:0;topk_bilstm_moe/sequential_407/dense_592/Relu:activations:0*
N*
T0*&
_output_shapes
:*

axisР
'topk_bilstm_moe/lambda_98/einsum/EinsumEinsum*topk_bilstm_moe/tf.stack_93/stack:output:00topk_bilstm_moe/lambda_96/einsum/Einsum:output:0*
N*
T0*"
_output_shapes
:*
equationabcd,ace->acdІ
6topk_bilstm_moe/bidirectional_92/forward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         ј
Dtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: љ
Ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:љ
Ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
>topk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_sliceStridedSlice?topk_bilstm_moe/bidirectional_92/forward_lstm_92/Shape:output:0Mtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice/stack:output:0Otopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice/stack_1:output:0Otopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЂ
?topk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :є
=topk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros/packedPackGtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice:output:0Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Ђ
<topk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ш
6topk_bilstm_moe/bidirectional_92/forward_lstm_92/zerosFillFtopk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros/packed:output:0Etopk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:Ѓ
Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :і
?topk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros_1/packedPackGtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice:output:0Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Ѓ
>topk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ч
8topk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros_1FillHtopk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros_1/packed:output:0Gtopk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:ћ
?topk_bilstm_moe/bidirectional_92/forward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ­
:topk_bilstm_moe/bidirectional_92/forward_lstm_92/transpose	Transpose0topk_bilstm_moe/lambda_98/einsum/Einsum:output:0Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:Ї
8topk_bilstm_moe/bidirectional_92/forward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         љ
Ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: њ
Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:њ
Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
@topk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1StridedSliceAtopk_bilstm_moe/bidirectional_92/forward_lstm_92/Shape_1:output:0Otopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1/stack:output:0Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1/stack_1:output:0Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskЌ
Ltopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         К
>topk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2TensorListReserveUtopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2/element_shape:output:0Itopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУми
ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      з
Xtopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor>topk_bilstm_moe/bidirectional_92/forward_lstm_92/transpose:y:0otopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмљ
Ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: њ
Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:њ
Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
@topk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_2StridedSlice>topk_bilstm_moe/bidirectional_92/forward_lstm_92/transpose:y:0Otopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_2/stack:output:0Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_2/stack_1:output:0Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskЖ
Ptopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOpYtopk_bilstm_moe_bidirectional_92_forward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ў
Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMulMatMulItopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_2:output:0Xtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: Ь
Rtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp[topk_bilstm_moe_bidirectional_92_forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Њ
Ctopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul_1MatMul?topk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros:output:0Ztopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: ї
>topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/addAddV2Ktopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul:product:0Mtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: У
Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpZtopk_bilstm_moe_bidirectional_92_forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ћ
Btopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/BiasAddBiasAddBtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/add:z:0Ytopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: ї
Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :├
@topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/splitSplitStopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/split/split_dim:output:0Ktopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split┴
Btopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/SigmoidSigmoidItopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:├
Dtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/Sigmoid_1SigmoidItopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:ч
>topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/mulMulHtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/Sigmoid_1:y:0Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:╗
?topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/ReluReluItopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:Є
@topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/mul_1MulFtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/Sigmoid:y:0Mtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Ч
@topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/add_1AddV2Btopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/mul:z:0Dtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:├
Dtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/Sigmoid_2SigmoidItopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:И
Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/Relu_1ReluDtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:І
@topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/mul_2MulHtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/Sigmoid_2:y:0Otopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:Ъ
Ntopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ╦
@topk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2_1TensorListReserveWtopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2_1/element_shape:output:0Itopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмw
5topk_bilstm_moe/bidirectional_92/forward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : ћ
Itopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         Ё
Ctopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ж

6topk_bilstm_moe/bidirectional_92/forward_lstm_92/whileWhileLtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/loop_counter:output:0Rtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/maximum_iterations:output:0>topk_bilstm_moe/bidirectional_92/forward_lstm_92/time:output:0Itopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2_1:handle:0?topk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros:output:0Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/zeros_1:output:0Itopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1:output:0htopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ytopk_bilstm_moe_bidirectional_92_forward_lstm_92_lstm_cell_matmul_readvariableop_resource[topk_bilstm_moe_bidirectional_92_forward_lstm_92_lstm_cell_matmul_1_readvariableop_resourceZtopk_bilstm_moe_bidirectional_92_forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_body_55808268*P
condHRF
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_cond_55808267*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations ▓
atopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ╠
Stopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStack?topk_bilstm_moe/bidirectional_92/forward_lstm_92/while:output:3jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0Ў
Ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         њ
Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: њ
Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
@topk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_3StridedSlice\topk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0Otopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_3/stack:output:0Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_3/stack_1:output:0Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskќ
Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          а
<topk_bilstm_moe/bidirectional_92/forward_lstm_92/transpose_1	Transpose\topk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:ї
8topk_bilstm_moe/bidirectional_92/forward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ї
7topk_bilstm_moe/bidirectional_92/backward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ј
Etopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Љ
Gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Љ
Gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╦
?topk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_sliceStridedSlice@topk_bilstm_moe/bidirectional_92/backward_lstm_92/Shape:output:0Ntopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice/stack:output:0Ptopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice/stack_1:output:0Ptopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskѓ
@topk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ѕ
>topk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros/packedPackHtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice:output:0Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:ѓ
=topk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    щ
7topk_bilstm_moe/bidirectional_92/backward_lstm_92/zerosFillGtopk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros/packed:output:0Ftopk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:ё
Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ї
@topk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros_1/packedPackHtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice:output:0Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:ё
?topk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *     
9topk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros_1FillItopk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros_1/packed:output:0Htopk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:Ћ
@topk_bilstm_moe/bidirectional_92/backward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ы
;topk_bilstm_moe/bidirectional_92/backward_lstm_92/transpose	Transpose0topk_bilstm_moe/lambda_98/einsum/Einsum:output:0Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:ј
9topk_bilstm_moe/bidirectional_92/backward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         Љ
Gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Њ
Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Њ
Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
Atopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1StridedSliceBtopk_bilstm_moe/bidirectional_92/backward_lstm_92/Shape_1:output:0Ptopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1/stack:output:0Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1/stack_1:output:0Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskў
Mtopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
?topk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2TensorListReserveVtopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2/element_shape:output:0Jtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмі
@topk_bilstm_moe/bidirectional_92/backward_lstm_92/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: Ђ
;topk_bilstm_moe/bidirectional_92/backward_lstm_92/ReverseV2	ReverseV2?topk_bilstm_moe/bidirectional_92/backward_lstm_92/transpose:y:0Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/ReverseV2/axis:output:0*
T0*"
_output_shapes
:И
gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ч
Ytopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorDtopk_bilstm_moe/bidirectional_92/backward_lstm_92/ReverseV2:output:0ptopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмЉ
Gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: Њ
Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Њ
Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┌
Atopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_2StridedSlice?topk_bilstm_moe/bidirectional_92/backward_lstm_92/transpose:y:0Ptopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_2/stack:output:0Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_2/stack_1:output:0Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskВ
Qtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOpZtopk_bilstm_moe_bidirectional_92_backward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0ю
Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMulMatMulJtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_2:output:0Ytopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ­
Stopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp\topk_bilstm_moe_bidirectional_92_backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0ќ
Dtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul_1MatMul@topk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros:output:0[topk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: Ј
?topk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/addAddV2Ltopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul:product:0Ntopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: Ж
Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp[topk_bilstm_moe_bidirectional_92_backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
Ctopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/BiasAddBiasAddCtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/add:z:0Ztopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Ї
Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
Atopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/splitSplitTtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/split/split_dim:output:0Ltopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split├
Ctopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/SigmoidSigmoidJtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:┼
Etopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/Sigmoid_1SigmoidJtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:■
?topk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/mulMulItopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/Sigmoid_1:y:0Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:й
@topk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/ReluReluJtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:і
Atopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/mul_1MulGtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/Sigmoid:y:0Ntopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

: 
Atopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/add_1AddV2Ctopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/mul:z:0Etopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:┼
Etopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/Sigmoid_2SigmoidJtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:║
Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/Relu_1ReluEtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:ј
Atopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/mul_2MulItopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/Sigmoid_2:y:0Ptopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:а
Otopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ╬
Atopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2_1TensorListReserveXtopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2_1/element_shape:output:0Jtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмx
6topk_bilstm_moe/bidirectional_92/backward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : Ћ
Jtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         є
Dtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э

7topk_bilstm_moe/bidirectional_92/backward_lstm_92/whileWhileMtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/loop_counter:output:0Stopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/maximum_iterations:output:0?topk_bilstm_moe/bidirectional_92/backward_lstm_92/time:output:0Jtopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2_1:handle:0@topk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros:output:0Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/zeros_1:output:0Jtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1:output:0itopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ztopk_bilstm_moe_bidirectional_92_backward_lstm_92_lstm_cell_matmul_readvariableop_resource\topk_bilstm_moe_bidirectional_92_backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource[topk_bilstm_moe_bidirectional_92_backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_body_55808409*Q
condIRG
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_cond_55808408*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations │
btopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ¤
Ttopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStack@topk_bilstm_moe/bidirectional_92/backward_lstm_92/while:output:3ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0џ
Gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         Њ
Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Њ
Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Э
Atopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_3StridedSlice]topk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0Ptopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_3/stack:output:0Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_3/stack_1:output:0Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskЌ
Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Б
=topk_bilstm_moe/bidirectional_92/backward_lstm_92/transpose_1	Transpose]topk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:Ї
9topk_bilstm_moe/bidirectional_92/backward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    y
/topk_bilstm_moe/bidirectional_92/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:р
*topk_bilstm_moe/bidirectional_92/ReverseV2	ReverseV2Atopk_bilstm_moe/bidirectional_92/backward_lstm_92/transpose_1:y:08topk_bilstm_moe/bidirectional_92/ReverseV2/axis:output:0*
T0*"
_output_shapes
:n
,topk_bilstm_moe/bidirectional_92/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ќ
'topk_bilstm_moe/bidirectional_92/concatConcatV2@topk_bilstm_moe/bidirectional_92/forward_lstm_92/transpose_1:y:03topk_bilstm_moe/bidirectional_92/ReverseV2:output:05topk_bilstm_moe/bidirectional_92/concat/axis:output:0*
N*
T0*"
_output_shapes
:ј
#topk_bilstm_moe/dropout_92/IdentityIdentity0topk_bilstm_moe/bidirectional_92/concat:output:0*
T0*"
_output_shapes
:q
 topk_bilstm_moe/flatten_92/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  ░
"topk_bilstm_moe/flatten_92/ReshapeReshape,topk_bilstm_moe/dropout_92/Identity:output:0)topk_bilstm_moe/flatten_92/Const:output:0*
T0*
_output_shapes
:	ђЕ
/topk_bilstm_moe/dense_593/MatMul/ReadVariableOpReadVariableOp8topk_bilstm_moe_dense_593_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0╣
 topk_bilstm_moe/dense_593/MatMulMatMul+topk_bilstm_moe/flatten_92/Reshape:output:07topk_bilstm_moe/dense_593/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:д
0topk_bilstm_moe/dense_593/BiasAdd/ReadVariableOpReadVariableOp9topk_bilstm_moe_dense_593_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
!topk_bilstm_moe/dense_593/BiasAddBiasAdd*topk_bilstm_moe/dense_593/MatMul:product:08topk_bilstm_moe/dense_593/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:p
IdentityIdentity*topk_bilstm_moe/dense_593/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:ш

NoOpNoOpS^topk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpR^topk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul/ReadVariableOpT^topk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp8^topk_bilstm_moe/bidirectional_92/backward_lstm_92/whileR^topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpQ^topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpS^topk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp7^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while1^topk_bilstm_moe/dense_576/BiasAdd/ReadVariableOp3^topk_bilstm_moe/dense_576/Tensordot/ReadVariableOp1^topk_bilstm_moe/dense_593/BiasAdd/ReadVariableOp0^topk_bilstm_moe/dense_593/MatMul/ReadVariableOp@^topk_bilstm_moe/sequential_392/dense_577/BiasAdd/ReadVariableOpB^topk_bilstm_moe/sequential_392/dense_577/Tensordot/ReadVariableOp@^topk_bilstm_moe/sequential_397/dense_582/BiasAdd/ReadVariableOpB^topk_bilstm_moe/sequential_397/dense_582/Tensordot/ReadVariableOp@^topk_bilstm_moe/sequential_402/dense_587/BiasAdd/ReadVariableOpB^topk_bilstm_moe/sequential_402/dense_587/Tensordot/ReadVariableOp@^topk_bilstm_moe/sequential_407/dense_592/BiasAdd/ReadVariableOpB^topk_bilstm_moe/sequential_407/dense_592/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:: : : : : : : : : : : : : : : : : : 2е
Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpRtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2д
Qtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul/ReadVariableOpQtopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul/ReadVariableOp2ф
Stopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpStopk_bilstm_moe/bidirectional_92/backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2r
7topk_bilstm_moe/bidirectional_92/backward_lstm_92/while7topk_bilstm_moe/bidirectional_92/backward_lstm_92/while2д
Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpQtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2ц
Ptopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpPtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2е
Rtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpRtopk_bilstm_moe/bidirectional_92/forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2p
6topk_bilstm_moe/bidirectional_92/forward_lstm_92/while6topk_bilstm_moe/bidirectional_92/forward_lstm_92/while2d
0topk_bilstm_moe/dense_576/BiasAdd/ReadVariableOp0topk_bilstm_moe/dense_576/BiasAdd/ReadVariableOp2h
2topk_bilstm_moe/dense_576/Tensordot/ReadVariableOp2topk_bilstm_moe/dense_576/Tensordot/ReadVariableOp2d
0topk_bilstm_moe/dense_593/BiasAdd/ReadVariableOp0topk_bilstm_moe/dense_593/BiasAdd/ReadVariableOp2b
/topk_bilstm_moe/dense_593/MatMul/ReadVariableOp/topk_bilstm_moe/dense_593/MatMul/ReadVariableOp2ѓ
?topk_bilstm_moe/sequential_392/dense_577/BiasAdd/ReadVariableOp?topk_bilstm_moe/sequential_392/dense_577/BiasAdd/ReadVariableOp2є
Atopk_bilstm_moe/sequential_392/dense_577/Tensordot/ReadVariableOpAtopk_bilstm_moe/sequential_392/dense_577/Tensordot/ReadVariableOp2ѓ
?topk_bilstm_moe/sequential_397/dense_582/BiasAdd/ReadVariableOp?topk_bilstm_moe/sequential_397/dense_582/BiasAdd/ReadVariableOp2є
Atopk_bilstm_moe/sequential_397/dense_582/Tensordot/ReadVariableOpAtopk_bilstm_moe/sequential_397/dense_582/Tensordot/ReadVariableOp2ѓ
?topk_bilstm_moe/sequential_402/dense_587/BiasAdd/ReadVariableOp?topk_bilstm_moe/sequential_402/dense_587/BiasAdd/ReadVariableOp2є
Atopk_bilstm_moe/sequential_402/dense_587/Tensordot/ReadVariableOpAtopk_bilstm_moe/sequential_402/dense_587/Tensordot/ReadVariableOp2ѓ
?topk_bilstm_moe/sequential_407/dense_592/BiasAdd/ReadVariableOp?topk_bilstm_moe/sequential_407/dense_592/BiasAdd/ReadVariableOp2є
Atopk_bilstm_moe/sequential_407/dense_592/Tensordot/ReadVariableOpAtopk_bilstm_moe/sequential_407/dense_592/Tensordot/ReadVariableOp:K G
"
_output_shapes
:
!
_user_specified_name	input_3:($
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
┘ќ
Ј
!__inference__traced_save_55814318
file_prefix9
'read_disablecopyonread_dense_576_kernel:5
'read_1_disablecopyonread_dense_576_bias:<
)read_2_disablecopyonread_dense_593_kernel:	ђ5
'read_3_disablecopyonread_dense_593_bias:;
)read_4_disablecopyonread_dense_577_kernel:5
'read_5_disablecopyonread_dense_577_bias:;
)read_6_disablecopyonread_dense_582_kernel:5
'read_7_disablecopyonread_dense_582_bias:;
)read_8_disablecopyonread_dense_587_kernel:5
'read_9_disablecopyonread_dense_587_bias:<
*read_10_disablecopyonread_dense_592_kernel:6
(read_11_disablecopyonread_dense_592_bias:]
Kread_12_disablecopyonread_bidirectional_92_forward_lstm_92_lstm_cell_kernel: g
Uread_13_disablecopyonread_bidirectional_92_forward_lstm_92_lstm_cell_recurrent_kernel: W
Iread_14_disablecopyonread_bidirectional_92_forward_lstm_92_lstm_cell_bias: ^
Lread_15_disablecopyonread_bidirectional_92_backward_lstm_92_lstm_cell_kernel: h
Vread_16_disablecopyonread_bidirectional_92_backward_lstm_92_lstm_cell_recurrent_kernel: X
Jread_17_disablecopyonread_bidirectional_92_backward_lstm_92_lstm_cell_bias: 
savev2_const
identity_37ѕбMergeV2CheckpointsбRead/DisableCopyOnReadбRead/ReadVariableOpбRead_1/DisableCopyOnReadбRead_1/ReadVariableOpбRead_10/DisableCopyOnReadбRead_10/ReadVariableOpбRead_11/DisableCopyOnReadбRead_11/ReadVariableOpбRead_12/DisableCopyOnReadбRead_12/ReadVariableOpбRead_13/DisableCopyOnReadбRead_13/ReadVariableOpбRead_14/DisableCopyOnReadбRead_14/ReadVariableOpбRead_15/DisableCopyOnReadбRead_15/ReadVariableOpбRead_16/DisableCopyOnReadбRead_16/ReadVariableOpбRead_17/DisableCopyOnReadбRead_17/ReadVariableOpбRead_2/DisableCopyOnReadбRead_2/ReadVariableOpбRead_3/DisableCopyOnReadбRead_3/ReadVariableOpбRead_4/DisableCopyOnReadбRead_4/ReadVariableOpбRead_5/DisableCopyOnReadбRead_5/ReadVariableOpбRead_6/DisableCopyOnReadбRead_6/ReadVariableOpбRead_7/DisableCopyOnReadбRead_7/ReadVariableOpбRead_8/DisableCopyOnReadбRead_8/ReadVariableOpбRead_9/DisableCopyOnReadбRead_9/ReadVariableOpw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_dense_576_kernel"/device:CPU:0*
_output_shapes
 Б
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_dense_576_kernel^Read/DisableCopyOnRead"/device:CPU:0*
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
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_dense_576_bias"/device:CPU:0*
_output_shapes
 Б
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_dense_576_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
Read_2/DisableCopyOnReadDisableCopyOnRead)read_2_disablecopyonread_dense_593_kernel"/device:CPU:0*
_output_shapes
 ф
Read_2/ReadVariableOpReadVariableOp)read_2_disablecopyonread_dense_593_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђd

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђ{
Read_3/DisableCopyOnReadDisableCopyOnRead'read_3_disablecopyonread_dense_593_bias"/device:CPU:0*
_output_shapes
 Б
Read_3/ReadVariableOpReadVariableOp'read_3_disablecopyonread_dense_593_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
Read_4/DisableCopyOnReadDisableCopyOnRead)read_4_disablecopyonread_dense_577_kernel"/device:CPU:0*
_output_shapes
 Е
Read_4/ReadVariableOpReadVariableOp)read_4_disablecopyonread_dense_577_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
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
Read_5/DisableCopyOnReadDisableCopyOnRead'read_5_disablecopyonread_dense_577_bias"/device:CPU:0*
_output_shapes
 Б
Read_5/ReadVariableOpReadVariableOp'read_5_disablecopyonread_dense_577_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_dense_582_kernel"/device:CPU:0*
_output_shapes
 Е
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_dense_582_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
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
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_dense_582_bias"/device:CPU:0*
_output_shapes
 Б
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_dense_582_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
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
Read_8/DisableCopyOnReadDisableCopyOnRead)read_8_disablecopyonread_dense_587_kernel"/device:CPU:0*
_output_shapes
 Е
Read_8/ReadVariableOpReadVariableOp)read_8_disablecopyonread_dense_587_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
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
Read_9/DisableCopyOnReadDisableCopyOnRead'read_9_disablecopyonread_dense_587_bias"/device:CPU:0*
_output_shapes
 Б
Read_9/ReadVariableOpReadVariableOp'read_9_disablecopyonread_dense_587_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
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
Read_10/DisableCopyOnReadDisableCopyOnRead*read_10_disablecopyonread_dense_592_kernel"/device:CPU:0*
_output_shapes
 г
Read_10/ReadVariableOpReadVariableOp*read_10_disablecopyonread_dense_592_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
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
Read_11/DisableCopyOnReadDisableCopyOnRead(read_11_disablecopyonread_dense_592_bias"/device:CPU:0*
_output_shapes
 д
Read_11/ReadVariableOpReadVariableOp(read_11_disablecopyonread_dense_592_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
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
:а
Read_12/DisableCopyOnReadDisableCopyOnReadKread_12_disablecopyonread_bidirectional_92_forward_lstm_92_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ═
Read_12/ReadVariableOpReadVariableOpKread_12_disablecopyonread_bidirectional_92_forward_lstm_92_lstm_cell_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
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

: ф
Read_13/DisableCopyOnReadDisableCopyOnReadUread_13_disablecopyonread_bidirectional_92_forward_lstm_92_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 О
Read_13/ReadVariableOpReadVariableOpUread_13_disablecopyonread_bidirectional_92_forward_lstm_92_lstm_cell_recurrent_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
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

: ъ
Read_14/DisableCopyOnReadDisableCopyOnReadIread_14_disablecopyonread_bidirectional_92_forward_lstm_92_lstm_cell_bias"/device:CPU:0*
_output_shapes
 К
Read_14/ReadVariableOpReadVariableOpIread_14_disablecopyonread_bidirectional_92_forward_lstm_92_lstm_cell_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
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
: А
Read_15/DisableCopyOnReadDisableCopyOnReadLread_15_disablecopyonread_bidirectional_92_backward_lstm_92_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ╬
Read_15/ReadVariableOpReadVariableOpLread_15_disablecopyonread_bidirectional_92_backward_lstm_92_lstm_cell_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
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

: Ф
Read_16/DisableCopyOnReadDisableCopyOnReadVread_16_disablecopyonread_bidirectional_92_backward_lstm_92_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 п
Read_16/ReadVariableOpReadVariableOpVread_16_disablecopyonread_bidirectional_92_backward_lstm_92_lstm_cell_recurrent_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
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

: Ъ
Read_17/DisableCopyOnReadDisableCopyOnReadJread_17_disablecopyonread_bidirectional_92_backward_lstm_92_lstm_cell_bias"/device:CPU:0*
_output_shapes
 ╚
Read_17/ReadVariableOpReadVariableOpJread_17_disablecopyonread_bidirectional_92_backward_lstm_92_lstm_cell_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
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
: ю
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┼
value╗BИB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЊ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ы
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
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
: ┘
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
_user_specified_namedense_576/kernel:.*
(
_user_specified_namedense_576/bias:0,
*
_user_specified_namedense_593/kernel:.*
(
_user_specified_namedense_593/bias:0,
*
_user_specified_namedense_577/kernel:.*
(
_user_specified_namedense_577/bias:0,
*
_user_specified_namedense_582/kernel:.*
(
_user_specified_namedense_582/bias:0	,
*
_user_specified_namedense_587/kernel:.
*
(
_user_specified_namedense_587/bias:0,
*
_user_specified_namedense_592/kernel:.*
(
_user_specified_namedense_592/bias:QM
K
_user_specified_name31bidirectional_92/forward_lstm_92/lstm_cell/kernel:[W
U
_user_specified_name=;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel:OK
I
_user_specified_name1/bidirectional_92/forward_lstm_92/lstm_cell/bias:RN
L
_user_specified_name42bidirectional_92/backward_lstm_92/lstm_cell/kernel:\X
V
_user_specified_name><bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel:PL
J
_user_specified_name20bidirectional_92/backward_lstm_92/lstm_cell/bias:=9

_output_shapes
: 

_user_specified_nameConst
╠	
═
while_cond_55813617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55813617___redundant_placeholder06
2while_while_cond_55813617___redundant_placeholder16
2while_while_cond_55813617___redundant_placeholder26
2while_while_cond_55813617___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
Ф
ѓ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814058

inputs
states_0
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:QM
'
_output_shapes
:         
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
ХJ
▒
#forward_lstm_92_while_body_55812306<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3;
7forward_lstm_92_while_forward_lstm_92_strided_slice_1_0w
sforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_92_while_identity$
 forward_lstm_92_while_identity_1$
 forward_lstm_92_while_identity_2$
 forward_lstm_92_while_identity_3$
 forward_lstm_92_while_identity_4$
 forward_lstm_92_while_identity_59
5forward_lstm_92_while_forward_lstm_92_strided_slice_1u
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpў
Gforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ь
9forward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_92_while_placeholderPforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0Х
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0┌
&forward_lstm_92/while/lstm_cell/MatMulMatMul@forward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ║
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0┴
(forward_lstm_92/while/lstm_cell/MatMul_1MatMul#forward_lstm_92_while_placeholder_2?forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: ╗
#forward_lstm_92/while/lstm_cell/addAddV20forward_lstm_92/while/lstm_cell/MatMul:product:02forward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: ┤
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0─
'forward_lstm_92/while/lstm_cell/BiasAddBiasAdd'forward_lstm_92/while/lstm_cell/add:z:0>forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
/forward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
%forward_lstm_92/while/lstm_cell/splitSplit8forward_lstm_92/while/lstm_cell/split/split_dim:output:00forward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitІ
'forward_lstm_92/while/lstm_cell/SigmoidSigmoid.forward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:Ї
)forward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:Д
#forward_lstm_92/while/lstm_cell/mulMul-forward_lstm_92/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:Ё
$forward_lstm_92/while/lstm_cell/ReluRelu.forward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:Х
%forward_lstm_92/while/lstm_cell/mul_1Mul+forward_lstm_92/while/lstm_cell/Sigmoid:y:02forward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Ф
%forward_lstm_92/while/lstm_cell/add_1AddV2'forward_lstm_92/while/lstm_cell/mul:z:0)forward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ї
)forward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:ѓ
&forward_lstm_92/while/lstm_cell/Relu_1Relu)forward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:║
%forward_lstm_92/while/lstm_cell/mul_2Mul-forward_lstm_92/while/lstm_cell/Sigmoid_2:y:04forward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:ѓ
:forward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_92_while_placeholder_1!forward_lstm_92_while_placeholder)forward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм]
forward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ї
forward_lstm_92/while/addAddV2!forward_lstm_92_while_placeholder$forward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/while/add_1AddV28forward_lstm_92_while_forward_lstm_92_while_loop_counter&forward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: Ѕ
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: ф
 forward_lstm_92/while/Identity_1Identity>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ѕ
 forward_lstm_92/while/Identity_2Identityforward_lstm_92/while/add:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Х
 forward_lstm_92/while/Identity_3IdentityJforward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ю
 forward_lstm_92/while/Identity_4Identity)forward_lstm_92/while/lstm_cell/mul_2:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes

:Ю
 forward_lstm_92/while/Identity_5Identity)forward_lstm_92/while/lstm_cell/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes

:с
forward_lstm_92/while/NoOpNoOp7^forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_92_while_forward_lstm_92_strided_slice_17forward_lstm_92_while_forward_lstm_92_strided_slice_1_0"I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0"M
 forward_lstm_92_while_identity_1)forward_lstm_92/while/Identity_1:output:0"M
 forward_lstm_92_while_identity_2)forward_lstm_92/while/Identity_2:output:0"M
 forward_lstm_92_while_identity_3)forward_lstm_92/while/Identity_3:output:0"M
 forward_lstm_92_while_identity_4)forward_lstm_92/while/Identity_4:output:0"M
 forward_lstm_92_while_identity_5)forward_lstm_92/while/Identity_5:output:0"ё
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"є
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ѓ
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"У
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2p
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:
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
_user_specified_name!forward_lstm_92/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
ѕ
Ў
,__inference_dense_592_layer_call_fn_55812721

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_592_layer_call_and_return_conditional_losses_55808767s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
55812715:($
"
_user_specified_name
55812717
Љ
I
-__inference_dropout_92_layer_call_fn_55812545

inputs
identity«
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
GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_92_layer_call_and_return_conditional_losses_55810983[
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
Г
Д
1__inference_sequential_397_layer_call_fn_55808649
dense_582_input
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCalldense_582_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808631s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_582_input:($
"
_user_specified_name
55808643:($
"
_user_specified_name
55808645
ХJ
▒
#forward_lstm_92_while_body_55812018<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3;
7forward_lstm_92_while_forward_lstm_92_strided_slice_1_0w
sforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_92_while_identity$
 forward_lstm_92_while_identity_1$
 forward_lstm_92_while_identity_2$
 forward_lstm_92_while_identity_3$
 forward_lstm_92_while_identity_4$
 forward_lstm_92_while_identity_59
5forward_lstm_92_while_forward_lstm_92_strided_slice_1u
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpў
Gforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ь
9forward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_92_while_placeholderPforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0Х
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0┌
&forward_lstm_92/while/lstm_cell/MatMulMatMul@forward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ║
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0┴
(forward_lstm_92/while/lstm_cell/MatMul_1MatMul#forward_lstm_92_while_placeholder_2?forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: ╗
#forward_lstm_92/while/lstm_cell/addAddV20forward_lstm_92/while/lstm_cell/MatMul:product:02forward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: ┤
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0─
'forward_lstm_92/while/lstm_cell/BiasAddBiasAdd'forward_lstm_92/while/lstm_cell/add:z:0>forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
/forward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
%forward_lstm_92/while/lstm_cell/splitSplit8forward_lstm_92/while/lstm_cell/split/split_dim:output:00forward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitІ
'forward_lstm_92/while/lstm_cell/SigmoidSigmoid.forward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:Ї
)forward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:Д
#forward_lstm_92/while/lstm_cell/mulMul-forward_lstm_92/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:Ё
$forward_lstm_92/while/lstm_cell/ReluRelu.forward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:Х
%forward_lstm_92/while/lstm_cell/mul_1Mul+forward_lstm_92/while/lstm_cell/Sigmoid:y:02forward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Ф
%forward_lstm_92/while/lstm_cell/add_1AddV2'forward_lstm_92/while/lstm_cell/mul:z:0)forward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ї
)forward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:ѓ
&forward_lstm_92/while/lstm_cell/Relu_1Relu)forward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:║
%forward_lstm_92/while/lstm_cell/mul_2Mul-forward_lstm_92/while/lstm_cell/Sigmoid_2:y:04forward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:ѓ
:forward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_92_while_placeholder_1!forward_lstm_92_while_placeholder)forward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм]
forward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ї
forward_lstm_92/while/addAddV2!forward_lstm_92_while_placeholder$forward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/while/add_1AddV28forward_lstm_92_while_forward_lstm_92_while_loop_counter&forward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: Ѕ
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: ф
 forward_lstm_92/while/Identity_1Identity>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ѕ
 forward_lstm_92/while/Identity_2Identityforward_lstm_92/while/add:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Х
 forward_lstm_92/while/Identity_3IdentityJforward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ю
 forward_lstm_92/while/Identity_4Identity)forward_lstm_92/while/lstm_cell/mul_2:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes

:Ю
 forward_lstm_92/while/Identity_5Identity)forward_lstm_92/while/lstm_cell/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes

:с
forward_lstm_92/while/NoOpNoOp7^forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_92_while_forward_lstm_92_strided_slice_17forward_lstm_92_while_forward_lstm_92_strided_slice_1_0"I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0"M
 forward_lstm_92_while_identity_1)forward_lstm_92/while/Identity_1:output:0"M
 forward_lstm_92_while_identity_2)forward_lstm_92/while/Identity_2:output:0"M
 forward_lstm_92_while_identity_3)forward_lstm_92/while/Identity_3:output:0"M
 forward_lstm_92_while_identity_4)forward_lstm_92/while/Identity_4:output:0"M
 forward_lstm_92_while_identity_5)forward_lstm_92/while/Identity_5:output:0"ё
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"є
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ѓ
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"У
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2p
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:
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
_user_specified_name!forward_lstm_92/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
б	
й
3__inference_backward_lstm_92_layer_call_fn_55813412

inputs
unknown: 
	unknown_0: 
	unknown_1: 
identityѕбStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55810113|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_user_specified_name
55813404:($
"
_user_specified_name
55813406:($
"
_user_specified_name
55813408
й
Ї
#forward_lstm_92_while_cond_55811441<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3>
:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55811441___redundant_placeholder0V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55811441___redundant_placeholder1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55811441___redundant_placeholder2V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55811441___redundant_placeholder3"
forward_lstm_92_while_identity
б
forward_lstm_92/while/LessLess!forward_lstm_92_while_placeholder:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_92/strided_slice_1:

_output_shapes
:
Ў
Ї
#forward_lstm_92_while_cond_55812017<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3>
:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55812017___redundant_placeholder0V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55812017___redundant_placeholder1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55812017___redundant_placeholder2V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55812017___redundant_placeholder3"
forward_lstm_92_while_identity
б
forward_lstm_92/while/LessLess!forward_lstm_92_while_placeholder:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:
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
_user_specified_name!forward_lstm_92/strided_slice_1:

_output_shapes
:
лK
Л
$backward_lstm_92_while_body_55812447>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3=
9backward_lstm_92_while_backward_lstm_92_strided_slice_1_0y
ubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_92_while_identity%
!backward_lstm_92_while_identity_1%
!backward_lstm_92_while_identity_2%
!backward_lstm_92_while_identity_3%
!backward_lstm_92_while_identity_4%
!backward_lstm_92_while_identity_5;
7backward_lstm_92_while_backward_lstm_92_strided_slice_1w
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpЎ
Hbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ы
:backward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_92_while_placeholderQbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0И
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0П
'backward_lstm_92/while/lstm_cell/MatMulMatMulAbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ╝
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0─
)backward_lstm_92/while/lstm_cell/MatMul_1MatMul$backward_lstm_92_while_placeholder_2@backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: Й
$backward_lstm_92/while/lstm_cell/addAddV21backward_lstm_92/while/lstm_cell/MatMul:product:03backward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: Х
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0К
(backward_lstm_92/while/lstm_cell/BiasAddBiasAdd(backward_lstm_92/while/lstm_cell/add:z:0?backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: r
0backward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
&backward_lstm_92/while/lstm_cell/splitSplit9backward_lstm_92/while/lstm_cell/split/split_dim:output:01backward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitЇ
(backward_lstm_92/while/lstm_cell/SigmoidSigmoid/backward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:Ј
*backward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:ф
$backward_lstm_92/while/lstm_cell/mulMul.backward_lstm_92/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:Є
%backward_lstm_92/while/lstm_cell/ReluRelu/backward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:╣
&backward_lstm_92/while/lstm_cell/mul_1Mul,backward_lstm_92/while/lstm_cell/Sigmoid:y:03backward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:«
&backward_lstm_92/while/lstm_cell/add_1AddV2(backward_lstm_92/while/lstm_cell/mul:z:0*backward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ј
*backward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:ё
'backward_lstm_92/while/lstm_cell/Relu_1Relu*backward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:й
&backward_lstm_92/while/lstm_cell/mul_2Mul.backward_lstm_92/while/lstm_cell/Sigmoid_2:y:05backward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:є
;backward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_92_while_placeholder_1"backward_lstm_92_while_placeholder*backward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм^
backward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ј
backward_lstm_92/while/addAddV2"backward_lstm_92_while_placeholder%backward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
backward_lstm_92/while/add_1AddV2:backward_lstm_92_while_backward_lstm_92_while_loop_counter'backward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: ї
backward_lstm_92/while/IdentityIdentity backward_lstm_92/while/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: «
!backward_lstm_92/while/Identity_1Identity@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ї
!backward_lstm_92/while/Identity_2Identitybackward_lstm_92/while/add:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ╣
!backward_lstm_92/while/Identity_3IdentityKbackward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: а
!backward_lstm_92/while/Identity_4Identity*backward_lstm_92/while/lstm_cell/mul_2:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes

:а
!backward_lstm_92/while/Identity_5Identity*backward_lstm_92/while/lstm_cell/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes

:у
backward_lstm_92/while/NoOpNoOp8^backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_92_while_backward_lstm_92_strided_slice_19backward_lstm_92_while_backward_lstm_92_strided_slice_1_0"K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0"O
!backward_lstm_92_while_identity_1*backward_lstm_92/while/Identity_1:output:0"O
!backward_lstm_92_while_identity_2*backward_lstm_92/while/Identity_2:output:0"O
!backward_lstm_92_while_identity_3*backward_lstm_92/while/Identity_3:output:0"O
!backward_lstm_92_while_identity_4*backward_lstm_92/while/Identity_4:output:0"O
!backward_lstm_92_while_identity_5*backward_lstm_92/while/Identity_5:output:0"є
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"ѕ
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ё
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"В
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2r
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:
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
_user_specified_name" backward_lstm_92/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
Ў
Ї
#forward_lstm_92_while_cond_55810337<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3>
:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55810337___redundant_placeholder0V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55810337___redundant_placeholder1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55810337___redundant_placeholder2V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55810337___redundant_placeholder3"
forward_lstm_92_while_identity
б
forward_lstm_92/while/LessLess!forward_lstm_92_while_placeholder:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:
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
_user_specified_name!forward_lstm_92/strided_slice_1:

_output_shapes
:
№
s
G__inference_lambda_97_layer_call_and_return_conditional_losses_55811291
inputs_0
inputs_1
identityѓ
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
к
┴
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_cond_55808408ђ
|topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_loop_counterЄ
ѓtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_maximum_iterationsG
Ctopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholderI
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder_1I
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder_2I
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder_3ѓ
~topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_less_topk_bilstm_moe_bidirectional_92_backward_lstm_92_strided_slice_1Џ
ќtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_cond_55808408___redundant_placeholder0Џ
ќtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_cond_55808408___redundant_placeholder1Џ
ќtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_cond_55808408___redundant_placeholder2Џ
ќtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_cond_55808408___redundant_placeholder3D
@topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity
ф
<topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/LessLessCtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder~topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_less_topk_bilstm_moe_bidirectional_92_backward_lstm_92_strided_slice_1*
T0*
_output_shapes
: »
@topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/IdentityIdentity@topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "Ї
@topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identityItopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::| x

_output_shapes
: 
^
_user_specified_nameFDtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/loop_counter:ѓ~

_output_shapes
: 
d
_user_specified_nameLJtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/maximum_iterations:
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
_user_specified_nameCAtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1:

_output_shapes
:
Ф
ѓ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814156

inputs
states_0
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:QM
'
_output_shapes
:         
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
У8
▒
while_body_55810029
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        »
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
У8
▒
while_body_55813763
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        »
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
Ф
ѓ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814090

inputs
states_0
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:QM
'
_output_shapes
:         
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
И
А
$backward_lstm_92_while_cond_55812446>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3@
<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55812446___redundant_placeholder0X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55812446___redundant_placeholder1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55812446___redundant_placeholder2X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55812446___redundant_placeholder3#
backward_lstm_92_while_identity
д
backward_lstm_92/while/LessLess"backward_lstm_92_while_placeholder<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_92/while/IdentityIdentitybackward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:
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
_user_specified_name" backward_lstm_92/strided_slice_1:

_output_shapes
:
№
s
G__inference_lambda_97_layer_call_and_return_conditional_losses_55811285
inputs_0
inputs_1
identityѓ
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
У8
▒
while_body_55809719
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        »
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
еJ
љ
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55809961

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:Ы
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55809877*
condR
while_cond_55809876*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
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
Л	
щ
G__inference_dense_593_layer_call_and_return_conditional_losses_55812592

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
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
:	ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	ђ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ЉL
Љ
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813847

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: є
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        т
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:Ы
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55813763*
condR
while_cond_55813762*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
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
╠	
═
while_cond_55809876
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55809876___redundant_placeholder06
2while_while_cond_55809876___redundant_placeholder16
2while_while_cond_55809876___redundant_placeholder26
2while_while_cond_55809876___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
╠	
═
while_cond_55810028
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55810028___redundant_placeholder06
2while_while_cond_55810028___redundant_placeholder16
2while_while_cond_55810028___redundant_placeholder26
2while_while_cond_55810028___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
еJ
љ
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55809651

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:Ы
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55809567*
condR
while_cond_55809566*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
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
ѕ
Ў
,__inference_dense_582_layer_call_fn_55812641

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_582_layer_call_and_return_conditional_losses_55808615s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
55812635:($
"
_user_specified_name
55812637
І
I
-__inference_flatten_92_layer_call_fn_55812567

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	ђ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_flatten_92_layer_call_and_return_conditional_losses_55810599X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
Л	
щ
G__inference_dense_593_layer_call_and_return_conditional_losses_55810610

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
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
:	ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	ђ
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
й╣
ю
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55810965

inputsJ
8forward_lstm_92_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_92_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource: 
identityѕб1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpб2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбbackward_lstm_92/whileб0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpб1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбforward_lstm_92/whilej
forward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         m
#forward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
forward_lstm_92/strided_sliceStridedSliceforward_lstm_92/Shape:output:0,forward_lstm_92/strided_slice/stack:output:0.forward_lstm_92/strided_slice/stack_1:output:0.forward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Б
forward_lstm_92/zeros/packedPack&forward_lstm_92/strided_slice:output:0'forward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
forward_lstm_92/zerosFill%forward_lstm_92/zeros/packed:output:0$forward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:b
 forward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/zeros_1/packedPack&forward_lstm_92/strided_slice:output:0)forward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ў
forward_lstm_92/zeros_1Fill'forward_lstm_92/zeros_1/packed:output:0&forward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:s
forward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ё
forward_lstm_92/transpose	Transposeinputs'forward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:l
forward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         o
%forward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
forward_lstm_92/strided_slice_1StridedSlice forward_lstm_92/Shape_1:output:0.forward_lstm_92/strided_slice_1/stack:output:00forward_lstm_92/strided_slice_1/stack_1:output:00forward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         С
forward_lstm_92/TensorArrayV2TensorListReserve4forward_lstm_92/TensorArrayV2/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмќ
Eforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      љ
7forward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_92/transpose:y:0Nforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмo
%forward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
forward_lstm_92/strided_slice_2StridedSliceforward_lstm_92/transpose:y:0.forward_lstm_92/strided_slice_2/stack:output:00forward_lstm_92/strided_slice_2/stack_1:output:00forward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskе
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Х
 forward_lstm_92/lstm_cell/MatMulMatMul(forward_lstm_92/strided_slice_2:output:07forward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: г
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0░
"forward_lstm_92/lstm_cell/MatMul_1MatMulforward_lstm_92/zeros:output:09forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: Е
forward_lstm_92/lstm_cell/addAddV2*forward_lstm_92/lstm_cell/MatMul:product:0,forward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: д
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0▓
!forward_lstm_92/lstm_cell/BiasAddBiasAdd!forward_lstm_92/lstm_cell/add:z:08forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: k
)forward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Я
forward_lstm_92/lstm_cell/splitSplit2forward_lstm_92/lstm_cell/split/split_dim:output:0*forward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split
!forward_lstm_92/lstm_cell/SigmoidSigmoid(forward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:Ђ
#forward_lstm_92/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:ў
forward_lstm_92/lstm_cell/mulMul'forward_lstm_92/lstm_cell/Sigmoid_1:y:0 forward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:y
forward_lstm_92/lstm_cell/ReluRelu(forward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:ц
forward_lstm_92/lstm_cell/mul_1Mul%forward_lstm_92/lstm_cell/Sigmoid:y:0,forward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Ў
forward_lstm_92/lstm_cell/add_1AddV2!forward_lstm_92/lstm_cell/mul:z:0#forward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ђ
#forward_lstm_92/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:v
 forward_lstm_92/lstm_cell/Relu_1Relu#forward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:е
forward_lstm_92/lstm_cell/mul_2Mul'forward_lstm_92/lstm_cell/Sigmoid_2:y:0.forward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:~
-forward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      У
forward_lstm_92/TensorArrayV2_1TensorListReserve6forward_lstm_92/TensorArrayV2_1/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмV
forward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         d
"forward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ю
forward_lstm_92/whileWhile+forward_lstm_92/while/loop_counter:output:01forward_lstm_92/while/maximum_iterations:output:0forward_lstm_92/time:output:0(forward_lstm_92/TensorArrayV2_1:handle:0forward_lstm_92/zeros:output:0 forward_lstm_92/zeros_1:output:0(forward_lstm_92/strided_slice_1:output:0Gforward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_92_lstm_cell_matmul_readvariableop_resource:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
#forward_lstm_92_while_body_55810736*/
cond'R%
#forward_lstm_92_while_cond_55810735*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations Љ
@forward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ж
2forward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_92/while:output:3Iforward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0x
%forward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         q
'forward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
forward_lstm_92/strided_slice_3StridedSlice;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_92/strided_slice_3/stack:output:00forward_lstm_92/strided_slice_3/stack_1:output:00forward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masku
 forward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
forward_lstm_92/transpose_1	Transpose;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:k
forward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
backward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         n
$backward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
backward_lstm_92/strided_sliceStridedSlicebackward_lstm_92/Shape:output:0-backward_lstm_92/strided_slice/stack:output:0/backward_lstm_92/strided_slice/stack_1:output:0/backward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :д
backward_lstm_92/zeros/packedPack'backward_lstm_92/strided_slice:output:0(backward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ќ
backward_lstm_92/zerosFill&backward_lstm_92/zeros/packed:output:0%backward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:c
!backward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ф
backward_lstm_92/zeros_1/packedPack'backward_lstm_92/strided_slice:output:0*backward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
backward_lstm_92/zeros_1Fill(backward_lstm_92/zeros_1/packed:output:0'backward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:t
backward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          є
backward_lstm_92/transpose	Transposeinputs(backward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:m
backward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
&backward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 backward_lstm_92/strided_slice_1StridedSlice!backward_lstm_92/Shape_1:output:0/backward_lstm_92/strided_slice_1/stack:output:01backward_lstm_92/strided_slice_1/stack_1:output:01backward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         у
backward_lstm_92/TensorArrayV2TensorListReserve5backward_lstm_92/TensorArrayV2/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмi
backward_lstm_92/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ъ
backward_lstm_92/ReverseV2	ReverseV2backward_lstm_92/transpose:y:0(backward_lstm_92/ReverseV2/axis:output:0*
T0*"
_output_shapes
:Ќ
Fbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ў
8backward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_92/ReverseV2:output:0Obackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмp
&backward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
 backward_lstm_92/strided_slice_2StridedSlicebackward_lstm_92/transpose:y:0/backward_lstm_92/strided_slice_2/stack:output:01backward_lstm_92/strided_slice_2/stack_1:output:01backward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskф
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0╣
!backward_lstm_92/lstm_cell/MatMulMatMul)backward_lstm_92/strided_slice_2:output:08backward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: «
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0│
#backward_lstm_92/lstm_cell/MatMul_1MatMulbackward_lstm_92/zeros:output:0:backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: г
backward_lstm_92/lstm_cell/addAddV2+backward_lstm_92/lstm_cell/MatMul:product:0-backward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: е
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0х
"backward_lstm_92/lstm_cell/BiasAddBiasAdd"backward_lstm_92/lstm_cell/add:z:09backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: l
*backward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :с
 backward_lstm_92/lstm_cell/splitSplit3backward_lstm_92/lstm_cell/split/split_dim:output:0+backward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitЂ
"backward_lstm_92/lstm_cell/SigmoidSigmoid)backward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:Ѓ
$backward_lstm_92/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:Џ
backward_lstm_92/lstm_cell/mulMul(backward_lstm_92/lstm_cell/Sigmoid_1:y:0!backward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:{
backward_lstm_92/lstm_cell/ReluRelu)backward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:Д
 backward_lstm_92/lstm_cell/mul_1Mul&backward_lstm_92/lstm_cell/Sigmoid:y:0-backward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:ю
 backward_lstm_92/lstm_cell/add_1AddV2"backward_lstm_92/lstm_cell/mul:z:0$backward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ѓ
$backward_lstm_92/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:x
!backward_lstm_92/lstm_cell/Relu_1Relu$backward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:Ф
 backward_lstm_92/lstm_cell/mul_2Mul(backward_lstm_92/lstm_cell/Sigmoid_2:y:0/backward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:
.backward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      в
 backward_lstm_92/TensorArrayV2_1TensorListReserve7backward_lstm_92/TensorArrayV2_1/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмW
backward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         e
#backward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ф
backward_lstm_92/whileWhile,backward_lstm_92/while/loop_counter:output:02backward_lstm_92/while/maximum_iterations:output:0backward_lstm_92/time:output:0)backward_lstm_92/TensorArrayV2_1:handle:0backward_lstm_92/zeros:output:0!backward_lstm_92/zeros_1:output:0)backward_lstm_92/strided_slice_1:output:0Hbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_92_lstm_cell_matmul_readvariableop_resource;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
$backward_lstm_92_while_body_55810877*0
cond(R&
$backward_lstm_92_while_cond_55810876*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations њ
Abackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      В
3backward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_92/while:output:3Jbackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0y
&backward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         r
(backward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
 backward_lstm_92/strided_slice_3StridedSlice<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_92/strided_slice_3/stack:output:01backward_lstm_92/strided_slice_3/stack_1:output:01backward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskv
!backward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          └
backward_lstm_92/transpose_1	Transpose<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:l
backward_lstm_92/runtimeConst"/device:CPU:0*
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
	ReverseV2	ReverseV2 backward_lstm_92/transpose_1:y:0ReverseV2/axis:output:0*
T0*"
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Њ
concatConcatV2forward_lstm_92/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*"
_output_shapes
:Y
IdentityIdentityconcat:output:0^NoOp*
T0*"
_output_shapes
:ѕ
NoOpNoOp2^backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_92/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_92/while1^forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_92/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2f
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_92/whilebackward_lstm_92/while2d
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_92/whileforward_lstm_92/while:J F
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
Н
┼
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55809816

inputs*
forward_lstm_92_55809652: *
forward_lstm_92_55809654: &
forward_lstm_92_55809656: +
backward_lstm_92_55809804: +
backward_lstm_92_55809806: '
backward_lstm_92_55809808: 
identityѕб(backward_lstm_92/StatefulPartitionedCallб'forward_lstm_92/StatefulPartitionedCall╗
'forward_lstm_92/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_92_55809652forward_lstm_92_55809654forward_lstm_92_55809656*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55809651└
(backward_lstm_92/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_92_55809804backward_lstm_92_55809806backward_lstm_92_55809808*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55809803X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:А
	ReverseV2	ReverseV21backward_lstm_92/StatefulPartitionedCall:output:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Х
concatConcatV20forward_lstm_92/StatefulPartitionedCall:output:0ReverseV2:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  k
IdentityIdentityconcat:output:0^NoOp*
T0*4
_output_shapes"
 :                  w
NoOpNoOp)^backward_lstm_92/StatefulPartitionedCall(^forward_lstm_92/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 2T
(backward_lstm_92/StatefulPartitionedCall(backward_lstm_92/StatefulPartitionedCall2R
'forward_lstm_92/StatefulPartitionedCall'forward_lstm_92/StatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_user_specified_name
55809652:($
"
_user_specified_name
55809654:($
"
_user_specified_name
55809656:($
"
_user_specified_name
55809804:($
"
_user_specified_name
55809806:($
"
_user_specified_name
55809808
╝:
Є
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55809303

inputs$
lstm_cell_55809221: $
lstm_cell_55809223:  
lstm_cell_55809225: 
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       т
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЬ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55809221lstm_cell_55809223lstm_cell_55809225*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809220n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55809221lstm_cell_55809223lstm_cell_55809225*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55809234*
condR
while_cond_55809233*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:($
"
_user_specified_name
55809221:($
"
_user_specified_name
55809223:($
"
_user_specified_name
55809225
▀8
▒
while_body_55813618
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
ќ	
┐
3__inference_backward_lstm_92_layer_call_fn_55813390
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
identityѕбStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55809450|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55813382:($
"
_user_specified_name
55813384:($
"
_user_specified_name
55813386
а	
╝
2__inference_forward_lstm_92_layer_call_fn_55812796

inputs
unknown: 
	unknown_0: 
	unknown_1: 
identityѕбStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55809961|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_user_specified_name
55812788:($
"
_user_specified_name
55812790:($
"
_user_specified_name
55812792
╠	
═
while_cond_55809566
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55809566___redundant_placeholder06
2while_while_cond_55809566___redundant_placeholder16
2while_while_cond_55809566___redundant_placeholder26
2while_while_cond_55809566___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
К
f
H__inference_dropout_92_layer_call_and_return_conditional_losses_55812562

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
Н
┼
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55810126

inputs*
forward_lstm_92_55809962: *
forward_lstm_92_55809964: &
forward_lstm_92_55809966: +
backward_lstm_92_55810114: +
backward_lstm_92_55810116: '
backward_lstm_92_55810118: 
identityѕб(backward_lstm_92/StatefulPartitionedCallб'forward_lstm_92/StatefulPartitionedCall╗
'forward_lstm_92/StatefulPartitionedCallStatefulPartitionedCallinputsforward_lstm_92_55809962forward_lstm_92_55809964forward_lstm_92_55809966*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55809961└
(backward_lstm_92/StatefulPartitionedCallStatefulPartitionedCallinputsbackward_lstm_92_55810114backward_lstm_92_55810116backward_lstm_92_55810118*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55810113X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:А
	ReverseV2	ReverseV21backward_lstm_92/StatefulPartitionedCall:output:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Х
concatConcatV20forward_lstm_92/StatefulPartitionedCall:output:0ReverseV2:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  k
IdentityIdentityconcat:output:0^NoOp*
T0*4
_output_shapes"
 :                  w
NoOpNoOp)^backward_lstm_92/StatefulPartitionedCall(^forward_lstm_92/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 2T
(backward_lstm_92/StatefulPartitionedCall(backward_lstm_92/StatefulPartitionedCall2R
'forward_lstm_92/StatefulPartitionedCall'forward_lstm_92/StatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_user_specified_name
55809962:($
"
_user_specified_name
55809964:($
"
_user_specified_name
55809966:($
"
_user_specified_name
55810114:($
"
_user_specified_name
55810116:($
"
_user_specified_name
55810118
Ў
Ї
#forward_lstm_92_while_cond_55812305<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3>
:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55812305___redundant_placeholder0V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55812305___redundant_placeholder1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55812305___redundant_placeholder2V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55812305___redundant_placeholder3"
forward_lstm_92_while_identity
б
forward_lstm_92/while/LessLess!forward_lstm_92_while_placeholder:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:
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
_user_specified_name!forward_lstm_92/strided_slice_1:

_output_shapes
:
└
Ы
,__inference_lstm_cell_layer_call_fn_55814026

inputs
states_0
states_1
unknown: 
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2ѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809017o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:($
"
_user_specified_name
55814014:($
"
_user_specified_name
55814016:($
"
_user_specified_name
55814018
С
■
G__inference_dense_587_layer_call_and_return_conditional_losses_55808691

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
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
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : ю
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
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
╠	
═
while_cond_55813762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55813762___redundant_placeholder06
2while_while_cond_55813762___redundant_placeholder16
2while_while_cond_55813762___redundant_placeholder26
2while_while_cond_55813762___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
у
q
G__inference_lambda_97_layer_call_and_return_conditional_losses_55810644

inputs
inputs_1
identityђ
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
й╣
ю
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55810567

inputsJ
8forward_lstm_92_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_92_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource: 
identityѕб1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpб2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбbackward_lstm_92/whileб0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpб1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбforward_lstm_92/whilej
forward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         m
#forward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
forward_lstm_92/strided_sliceStridedSliceforward_lstm_92/Shape:output:0,forward_lstm_92/strided_slice/stack:output:0.forward_lstm_92/strided_slice/stack_1:output:0.forward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Б
forward_lstm_92/zeros/packedPack&forward_lstm_92/strided_slice:output:0'forward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
forward_lstm_92/zerosFill%forward_lstm_92/zeros/packed:output:0$forward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:b
 forward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/zeros_1/packedPack&forward_lstm_92/strided_slice:output:0)forward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ў
forward_lstm_92/zeros_1Fill'forward_lstm_92/zeros_1/packed:output:0&forward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:s
forward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ё
forward_lstm_92/transpose	Transposeinputs'forward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:l
forward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         o
%forward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
forward_lstm_92/strided_slice_1StridedSlice forward_lstm_92/Shape_1:output:0.forward_lstm_92/strided_slice_1/stack:output:00forward_lstm_92/strided_slice_1/stack_1:output:00forward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         С
forward_lstm_92/TensorArrayV2TensorListReserve4forward_lstm_92/TensorArrayV2/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмќ
Eforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      љ
7forward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_92/transpose:y:0Nforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмo
%forward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
forward_lstm_92/strided_slice_2StridedSliceforward_lstm_92/transpose:y:0.forward_lstm_92/strided_slice_2/stack:output:00forward_lstm_92/strided_slice_2/stack_1:output:00forward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskе
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Х
 forward_lstm_92/lstm_cell/MatMulMatMul(forward_lstm_92/strided_slice_2:output:07forward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: г
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0░
"forward_lstm_92/lstm_cell/MatMul_1MatMulforward_lstm_92/zeros:output:09forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: Е
forward_lstm_92/lstm_cell/addAddV2*forward_lstm_92/lstm_cell/MatMul:product:0,forward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: д
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0▓
!forward_lstm_92/lstm_cell/BiasAddBiasAdd!forward_lstm_92/lstm_cell/add:z:08forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: k
)forward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Я
forward_lstm_92/lstm_cell/splitSplit2forward_lstm_92/lstm_cell/split/split_dim:output:0*forward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split
!forward_lstm_92/lstm_cell/SigmoidSigmoid(forward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:Ђ
#forward_lstm_92/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:ў
forward_lstm_92/lstm_cell/mulMul'forward_lstm_92/lstm_cell/Sigmoid_1:y:0 forward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:y
forward_lstm_92/lstm_cell/ReluRelu(forward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:ц
forward_lstm_92/lstm_cell/mul_1Mul%forward_lstm_92/lstm_cell/Sigmoid:y:0,forward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Ў
forward_lstm_92/lstm_cell/add_1AddV2!forward_lstm_92/lstm_cell/mul:z:0#forward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ђ
#forward_lstm_92/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:v
 forward_lstm_92/lstm_cell/Relu_1Relu#forward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:е
forward_lstm_92/lstm_cell/mul_2Mul'forward_lstm_92/lstm_cell/Sigmoid_2:y:0.forward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:~
-forward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      У
forward_lstm_92/TensorArrayV2_1TensorListReserve6forward_lstm_92/TensorArrayV2_1/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмV
forward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         d
"forward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ю
forward_lstm_92/whileWhile+forward_lstm_92/while/loop_counter:output:01forward_lstm_92/while/maximum_iterations:output:0forward_lstm_92/time:output:0(forward_lstm_92/TensorArrayV2_1:handle:0forward_lstm_92/zeros:output:0 forward_lstm_92/zeros_1:output:0(forward_lstm_92/strided_slice_1:output:0Gforward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_92_lstm_cell_matmul_readvariableop_resource:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
#forward_lstm_92_while_body_55810338*/
cond'R%
#forward_lstm_92_while_cond_55810337*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations Љ
@forward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ж
2forward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_92/while:output:3Iforward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0x
%forward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         q
'forward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
forward_lstm_92/strided_slice_3StridedSlice;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_92/strided_slice_3/stack:output:00forward_lstm_92/strided_slice_3/stack_1:output:00forward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masku
 forward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
forward_lstm_92/transpose_1	Transpose;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:k
forward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
backward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         n
$backward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
backward_lstm_92/strided_sliceStridedSlicebackward_lstm_92/Shape:output:0-backward_lstm_92/strided_slice/stack:output:0/backward_lstm_92/strided_slice/stack_1:output:0/backward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :д
backward_lstm_92/zeros/packedPack'backward_lstm_92/strided_slice:output:0(backward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ќ
backward_lstm_92/zerosFill&backward_lstm_92/zeros/packed:output:0%backward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:c
!backward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ф
backward_lstm_92/zeros_1/packedPack'backward_lstm_92/strided_slice:output:0*backward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
backward_lstm_92/zeros_1Fill(backward_lstm_92/zeros_1/packed:output:0'backward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:t
backward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          є
backward_lstm_92/transpose	Transposeinputs(backward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:m
backward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
&backward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 backward_lstm_92/strided_slice_1StridedSlice!backward_lstm_92/Shape_1:output:0/backward_lstm_92/strided_slice_1/stack:output:01backward_lstm_92/strided_slice_1/stack_1:output:01backward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         у
backward_lstm_92/TensorArrayV2TensorListReserve5backward_lstm_92/TensorArrayV2/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмi
backward_lstm_92/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ъ
backward_lstm_92/ReverseV2	ReverseV2backward_lstm_92/transpose:y:0(backward_lstm_92/ReverseV2/axis:output:0*
T0*"
_output_shapes
:Ќ
Fbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ў
8backward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_92/ReverseV2:output:0Obackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмp
&backward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
 backward_lstm_92/strided_slice_2StridedSlicebackward_lstm_92/transpose:y:0/backward_lstm_92/strided_slice_2/stack:output:01backward_lstm_92/strided_slice_2/stack_1:output:01backward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskф
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0╣
!backward_lstm_92/lstm_cell/MatMulMatMul)backward_lstm_92/strided_slice_2:output:08backward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: «
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0│
#backward_lstm_92/lstm_cell/MatMul_1MatMulbackward_lstm_92/zeros:output:0:backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: г
backward_lstm_92/lstm_cell/addAddV2+backward_lstm_92/lstm_cell/MatMul:product:0-backward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: е
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0х
"backward_lstm_92/lstm_cell/BiasAddBiasAdd"backward_lstm_92/lstm_cell/add:z:09backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: l
*backward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :с
 backward_lstm_92/lstm_cell/splitSplit3backward_lstm_92/lstm_cell/split/split_dim:output:0+backward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitЂ
"backward_lstm_92/lstm_cell/SigmoidSigmoid)backward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:Ѓ
$backward_lstm_92/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:Џ
backward_lstm_92/lstm_cell/mulMul(backward_lstm_92/lstm_cell/Sigmoid_1:y:0!backward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:{
backward_lstm_92/lstm_cell/ReluRelu)backward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:Д
 backward_lstm_92/lstm_cell/mul_1Mul&backward_lstm_92/lstm_cell/Sigmoid:y:0-backward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:ю
 backward_lstm_92/lstm_cell/add_1AddV2"backward_lstm_92/lstm_cell/mul:z:0$backward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ѓ
$backward_lstm_92/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:x
!backward_lstm_92/lstm_cell/Relu_1Relu$backward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:Ф
 backward_lstm_92/lstm_cell/mul_2Mul(backward_lstm_92/lstm_cell/Sigmoid_2:y:0/backward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:
.backward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      в
 backward_lstm_92/TensorArrayV2_1TensorListReserve7backward_lstm_92/TensorArrayV2_1/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмW
backward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         e
#backward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ф
backward_lstm_92/whileWhile,backward_lstm_92/while/loop_counter:output:02backward_lstm_92/while/maximum_iterations:output:0backward_lstm_92/time:output:0)backward_lstm_92/TensorArrayV2_1:handle:0backward_lstm_92/zeros:output:0!backward_lstm_92/zeros_1:output:0)backward_lstm_92/strided_slice_1:output:0Hbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_92_lstm_cell_matmul_readvariableop_resource;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
$backward_lstm_92_while_body_55810479*0
cond(R&
$backward_lstm_92_while_cond_55810478*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations њ
Abackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      В
3backward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_92/while:output:3Jbackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0y
&backward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         r
(backward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
 backward_lstm_92/strided_slice_3StridedSlice<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_92/strided_slice_3/stack:output:01backward_lstm_92/strided_slice_3/stack_1:output:01backward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskv
!backward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          └
backward_lstm_92/transpose_1	Transpose<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:l
backward_lstm_92/runtimeConst"/device:CPU:0*
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
	ReverseV2	ReverseV2 backward_lstm_92/transpose_1:y:0ReverseV2/axis:output:0*
T0*"
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Њ
concatConcatV2forward_lstm_92/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*"
_output_shapes
:Y
IdentityIdentityconcat:output:0^NoOp*
T0*"
_output_shapes
:ѕ
NoOpNoOp2^backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_92/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_92/while1^forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_92/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2f
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_92/whilebackward_lstm_92/while2d
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_92/whileforward_lstm_92/while:J F
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
Д	
Я
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808774
dense_592_input$
dense_592_55808768: 
dense_592_55808770:
identityѕб!dense_592/StatefulPartitionedCallЄ
!dense_592/StatefulPartitionedCallStatefulPartitionedCalldense_592_inputdense_592_55808768dense_592_55808770*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_592_layer_call_and_return_conditional_losses_55808767}
IdentityIdentity*dense_592/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         F
NoOpNoOp"^dense_592/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 2F
!dense_592/StatefulPartitionedCall!dense_592/StatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_592_input:($
"
_user_specified_name
55808768:($
"
_user_specified_name
55808770
У8
▒
while_body_55809877
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        »
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
У8
▒
while_body_55809567
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        »
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
Й
▄
2__inference_topk_bilstm_moe_layer_call_fn_55811074
input_3
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

unknown_15:	ђ

unknown_16:
identityѕбStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8ѓ *V
fQRO
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810992f
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
_user_specified_name	input_3:($
"
_user_specified_name
55811036:($
"
_user_specified_name
55811038:($
"
_user_specified_name
55811040:($
"
_user_specified_name
55811042:($
"
_user_specified_name
55811044:($
"
_user_specified_name
55811046:($
"
_user_specified_name
55811048:($
"
_user_specified_name
55811050:(	$
"
_user_specified_name
55811052:(
$
"
_user_specified_name
55811054:($
"
_user_specified_name
55811056:($
"
_user_specified_name
55811058:($
"
_user_specified_name
55811060:($
"
_user_specified_name
55811062:($
"
_user_specified_name
55811064:($
"
_user_specified_name
55811066:($
"
_user_specified_name
55811068:($
"
_user_specified_name
55811070
П8
є
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55808955

inputs$
lstm_cell_55808873: $
lstm_cell_55808875:  
lstm_cell_55808877: 
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЬ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55808873lstm_cell_55808875lstm_cell_55808877*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55808872n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55808873lstm_cell_55808875lstm_cell_55808877*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55808886*
condR
while_cond_55808885*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:($
"
_user_specified_name
55808873:($
"
_user_specified_name
55808875:($
"
_user_specified_name
55808877
Т
q
G__inference_lambda_98_layer_call_and_return_conditional_losses_55810676

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
Ў
Ї
#forward_lstm_92_while_cond_55810735<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3>
:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55810735___redundant_placeholder0V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55810735___redundant_placeholder1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55810735___redundant_placeholder2V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55810735___redundant_placeholder3"
forward_lstm_92_while_identity
б
forward_lstm_92/while/LessLess!forward_lstm_92_while_placeholder:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:
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
_user_specified_name!forward_lstm_92/strided_slice_1:

_output_shapes
:
Т
q
G__inference_lambda_96_layer_call_and_return_conditional_losses_55810637

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
ћ	
Й
2__inference_forward_lstm_92_layer_call_fn_55812763
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55808955|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55812755:($
"
_user_specified_name
55812757:($
"
_user_specified_name
55812759
д
њ
3__inference_bidirectional_92_layer_call_fn_55811332
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55809816|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
=
_output_shapes+
):'                           
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55811318:($
"
_user_specified_name
55811320:($
"
_user_specified_name
55811322:($
"
_user_specified_name
55811324:($
"
_user_specified_name
55811326:($
"
_user_specified_name
55811328
вK
Њ
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813702
inputs_0:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       т
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55813618*
condR
while_cond_55813617*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
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
Г
Д
1__inference_sequential_407_layer_call_fn_55808792
dense_592_input
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCalldense_592_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808774s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_592_input:($
"
_user_specified_name
55808786:($
"
_user_specified_name
55808788
И
А
$backward_lstm_92_while_cond_55812158>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3@
<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55812158___redundant_placeholder0X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55812158___redundant_placeholder1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55812158___redundant_placeholder2X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55812158___redundant_placeholder3#
backward_lstm_92_while_identity
д
backward_lstm_92/while/LessLess"backward_lstm_92_while_placeholder<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_92/while/IdentityIdentitybackward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:
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
_user_specified_name" backward_lstm_92/strided_slice_1:

_output_shapes
:
ѕ
л
&__inference_signature_wrapper_55811219
input_3
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

unknown_15:	ђ

unknown_16:
identityѕбStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8ѓ *,
f'R%
#__inference__wrapped_model_55808506f
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
_user_specified_name	input_3:($
"
_user_specified_name
55811181:($
"
_user_specified_name
55811183:($
"
_user_specified_name
55811185:($
"
_user_specified_name
55811187:($
"
_user_specified_name
55811189:($
"
_user_specified_name
55811191:($
"
_user_specified_name
55811193:($
"
_user_specified_name
55811195:(	$
"
_user_specified_name
55811197:(
$
"
_user_specified_name
55811199:($
"
_user_specified_name
55811201:($
"
_user_specified_name
55811203:($
"
_user_specified_name
55811205:($
"
_user_specified_name
55811207:($
"
_user_specified_name
55811209:($
"
_user_specified_name
55811211:($
"
_user_specified_name
55811213:($
"
_user_specified_name
55811215
╠	
═
while_cond_55808885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55808885___redundant_placeholder06
2while_while_cond_55808885___redundant_placeholder16
2while_while_cond_55808885___redundant_placeholder26
2while_while_cond_55808885___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
їJ
њ
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55812939
inputs_0:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55812855*
condR
while_cond_55812854*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
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
С
■
G__inference_dense_577_layer_call_and_return_conditional_losses_55808539

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
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
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : ю
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
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
─s
Э
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_body_55808409ђ
|topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_loop_counterЄ
ѓtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_maximum_iterationsG
Ctopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholderI
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder_1I
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder_2I
Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder_3
{topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_strided_slice_1_0╝
иtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0t
btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: v
dtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: q
ctopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: D
@topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identityF
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_1F
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_2F
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_3F
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_4F
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_5}
ytopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_strided_slice_1║
хtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorr
`topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: t
btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: o
atopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕбXtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpбWtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpбYtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp║
itopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ў
[topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemиtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0Ctopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholderrtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0Щ
Wtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpbtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0└
Htopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMulMatMulbtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0_topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ■
Ytopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpdtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0Д
Jtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul_1MatMulEtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder_2atopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: А
Etopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/addAddV2Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul:product:0Ttopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: Э
Xtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpctopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0ф
Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/BiasAddBiasAddItopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/add:z:0`topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: Њ
Qtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :п
Gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/splitSplitZtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/split/split_dim:output:0Rtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split¤
Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/SigmoidSigmoidPtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:Л
Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/Sigmoid_1SigmoidPtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:Ї
Etopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/mulMulOtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/Sigmoid_1:y:0Etopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:╔
Ftopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/ReluReluPtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:ю
Gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/mul_1MulMtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/Sigmoid:y:0Ttopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Љ
Gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/add_1AddV2Itopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/mul:z:0Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Л
Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/Sigmoid_2SigmoidPtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:к
Htopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/Relu_1ReluKtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:а
Gtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/mul_2MulOtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/Sigmoid_2:y:0Vtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:і
\topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemEtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholder_1Ctopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholderKtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм
=topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ы
;topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/addAddV2Ctopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_placeholderFtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: Ђ
?topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
=topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/add_1AddV2|topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_loop_counterHtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: №
@topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/IdentityIdentityAtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/add_1:z:0=^topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/NoOp*
T0*
_output_shapes
: │
Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_1Identityѓtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_maximum_iterations=^topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/NoOp*
T0*
_output_shapes
: №
Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_2Identity?topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/add:z:0=^topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ю
Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_3Identityltopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0=^topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ѓ
Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_4IdentityKtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/mul_2:z:0=^topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/NoOp*
T0*
_output_shapes

:Ѓ
Btopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_5IdentityKtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/add_1:z:0=^topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/NoOp*
T0*
_output_shapes

:в
<topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/NoOpNoOpY^topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpX^topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpZ^topk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "Ї
@topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identityItopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity:output:0"Љ
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_1Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_1:output:0"Љ
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_2Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_2:output:0"Љ
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_3Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_3:output:0"Љ
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_4Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_4:output:0"Љ
Btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_identity_5Ktopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/Identity_5:output:0"╚
atopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourcectopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"╩
btopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourcedtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"к
`topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_readvariableop_resourcebtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"Ы
хtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorиtopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0"Э
ytopk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_strided_slice_1{topk_bilstm_moe_bidirectional_92_backward_lstm_92_while_topk_bilstm_moe_bidirectional_92_backward_lstm_92_strided_slice_1_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2┤
Xtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpXtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2▓
Wtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpWtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2Х
Ytopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpYtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:| x

_output_shapes
: 
^
_user_specified_nameFDtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/loop_counter:ѓ~

_output_shapes
: 
d
_user_specified_nameLJtopk_bilstm_moe/bidirectional_92/backward_lstm_92/while/maximum_iterations:
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
_user_specified_nameCAtopk_bilstm_moe/bidirectional_92/backward_lstm_92/strided_slice_1:њЇ

_output_shapes
: 
s
_user_specified_name[Ytopk_bilstm_moe/bidirectional_92/backward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
╠	
═
while_cond_55809718
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55809718___redundant_placeholder06
2while_while_cond_55809718___redundant_placeholder16
2while_while_cond_55809718___redundant_placeholder26
2while_while_cond_55809718___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
╠	
═
while_cond_55813140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55813140___redundant_placeholder06
2while_while_cond_55813140___redundant_placeholder16
2while_while_cond_55813140___redundant_placeholder26
2while_while_cond_55813140___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
С
■
G__inference_dense_582_layer_call_and_return_conditional_losses_55812672

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
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
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : ю
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
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
еJ
љ
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813368

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:Ы
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55813284*
condR
while_cond_55813283*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
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
Ф
ѓ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814188

inputs
states_0
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:QM
'
_output_shapes
:         
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
С$
о
while_body_55808886
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_55808910_0: ,
while_lstm_cell_55808912_0: (
while_lstm_cell_55808914_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_55808910: *
while_lstm_cell_55808912: &
while_lstm_cell_55808914: ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55808910_0while_lstm_cell_55808912_0while_lstm_cell_55808914_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55808872┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_55808910while_lstm_cell_55808910_0"6
while_lstm_cell_55808912while_lstm_cell_55808912_0"6
while_lstm_cell_55808914while_lstm_cell_55808914_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
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
:         :-)
'
_output_shapes
:         :GC
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
55808910:(	$
"
_user_specified_name
55808912:(
$
"
_user_specified_name
55808914
Г
Д
1__inference_sequential_402_layer_call_fn_55808725
dense_587_input
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCalldense_587_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808707s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_587_input:($
"
_user_specified_name
55808719:($
"
_user_specified_name
55808721
С$
о
while_body_55809031
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_55809055_0: ,
while_lstm_cell_55809057_0: (
while_lstm_cell_55809059_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_55809055: *
while_lstm_cell_55809057: &
while_lstm_cell_55809059: ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55809055_0while_lstm_cell_55809057_0while_lstm_cell_55809059_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809017┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_55809055while_lstm_cell_55809055_0"6
while_lstm_cell_55809057while_lstm_cell_55809057_0"6
while_lstm_cell_55809059while_lstm_cell_55809059_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
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
:         :-)
'
_output_shapes
:         :GC
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
55809055:(	$
"
_user_specified_name
55809057:(
$
"
_user_specified_name
55809059
їJ
њ
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813082
inputs_0:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55812998*
condR
while_cond_55812997*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
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
И
А
$backward_lstm_92_while_cond_55810876>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3@
<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55810876___redundant_placeholder0X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55810876___redundant_placeholder1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55810876___redundant_placeholder2X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55810876___redundant_placeholder3#
backward_lstm_92_while_identity
д
backward_lstm_92/while/LessLess"backward_lstm_92_while_placeholder<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_92/while/IdentityIdentitybackward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:
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
_user_specified_name" backward_lstm_92/strided_slice_1:

_output_shapes
:
Д	
Я
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808631
dense_582_input$
dense_582_55808625: 
dense_582_55808627:
identityѕб!dense_582/StatefulPartitionedCallЄ
!dense_582/StatefulPartitionedCallStatefulPartitionedCalldense_582_inputdense_582_55808625dense_582_55808627*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_582_layer_call_and_return_conditional_losses_55808615}
IdentityIdentity*dense_582/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         F
NoOpNoOp"^dense_582/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 2F
!dense_582/StatefulPartitionedCall!dense_582/StatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_582_input:($
"
_user_specified_name
55808625:($
"
_user_specified_name
55808627
┐
f
-__inference_dropout_92_layer_call_fn_55812540

inputs
identityѕбStatefulPartitionedCallЙ
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
GPU 2J 8ѓ *Q
fLRJ
H__inference_dropout_92_layer_call_and_return_conditional_losses_55810592j
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
Д	
Я
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808707
dense_587_input$
dense_587_55808701: 
dense_587_55808703:
identityѕб!dense_587/StatefulPartitionedCallЄ
!dense_587/StatefulPartitionedCallStatefulPartitionedCalldense_587_inputdense_587_55808701dense_587_55808703*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_587_layer_call_and_return_conditional_losses_55808691}
IdentityIdentity*dense_587/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         F
NoOpNoOp"^dense_587/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 2F
!dense_587/StatefulPartitionedCall!dense_587/StatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_587_input:($
"
_user_specified_name
55808701:($
"
_user_specified_name
55808703
С
■
G__inference_dense_592_layer_call_and_return_conditional_losses_55812752

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
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
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : ю
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
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
К
f
H__inference_dropout_92_layer_call_and_return_conditional_losses_55810983

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
╠	
═
while_cond_55813283
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55813283___redundant_placeholder06
2while_while_cond_55813283___redundant_placeholder16
2while_while_cond_55813283___redundant_placeholder26
2while_while_cond_55813283___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
лK
Л
$backward_lstm_92_while_body_55810877>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3=
9backward_lstm_92_while_backward_lstm_92_strided_slice_1_0y
ubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_92_while_identity%
!backward_lstm_92_while_identity_1%
!backward_lstm_92_while_identity_2%
!backward_lstm_92_while_identity_3%
!backward_lstm_92_while_identity_4%
!backward_lstm_92_while_identity_5;
7backward_lstm_92_while_backward_lstm_92_strided_slice_1w
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpЎ
Hbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ы
:backward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_92_while_placeholderQbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0И
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0П
'backward_lstm_92/while/lstm_cell/MatMulMatMulAbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ╝
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0─
)backward_lstm_92/while/lstm_cell/MatMul_1MatMul$backward_lstm_92_while_placeholder_2@backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: Й
$backward_lstm_92/while/lstm_cell/addAddV21backward_lstm_92/while/lstm_cell/MatMul:product:03backward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: Х
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0К
(backward_lstm_92/while/lstm_cell/BiasAddBiasAdd(backward_lstm_92/while/lstm_cell/add:z:0?backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: r
0backward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
&backward_lstm_92/while/lstm_cell/splitSplit9backward_lstm_92/while/lstm_cell/split/split_dim:output:01backward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitЇ
(backward_lstm_92/while/lstm_cell/SigmoidSigmoid/backward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:Ј
*backward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:ф
$backward_lstm_92/while/lstm_cell/mulMul.backward_lstm_92/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:Є
%backward_lstm_92/while/lstm_cell/ReluRelu/backward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:╣
&backward_lstm_92/while/lstm_cell/mul_1Mul,backward_lstm_92/while/lstm_cell/Sigmoid:y:03backward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:«
&backward_lstm_92/while/lstm_cell/add_1AddV2(backward_lstm_92/while/lstm_cell/mul:z:0*backward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ј
*backward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:ё
'backward_lstm_92/while/lstm_cell/Relu_1Relu*backward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:й
&backward_lstm_92/while/lstm_cell/mul_2Mul.backward_lstm_92/while/lstm_cell/Sigmoid_2:y:05backward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:є
;backward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_92_while_placeholder_1"backward_lstm_92_while_placeholder*backward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм^
backward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ј
backward_lstm_92/while/addAddV2"backward_lstm_92_while_placeholder%backward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
backward_lstm_92/while/add_1AddV2:backward_lstm_92_while_backward_lstm_92_while_loop_counter'backward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: ї
backward_lstm_92/while/IdentityIdentity backward_lstm_92/while/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: «
!backward_lstm_92/while/Identity_1Identity@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ї
!backward_lstm_92/while/Identity_2Identitybackward_lstm_92/while/add:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ╣
!backward_lstm_92/while/Identity_3IdentityKbackward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: а
!backward_lstm_92/while/Identity_4Identity*backward_lstm_92/while/lstm_cell/mul_2:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes

:а
!backward_lstm_92/while/Identity_5Identity*backward_lstm_92/while/lstm_cell/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes

:у
backward_lstm_92/while/NoOpNoOp8^backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_92_while_backward_lstm_92_strided_slice_19backward_lstm_92_while_backward_lstm_92_strided_slice_1_0"K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0"O
!backward_lstm_92_while_identity_1*backward_lstm_92/while/Identity_1:output:0"O
!backward_lstm_92_while_identity_2*backward_lstm_92/while/Identity_2:output:0"O
!backward_lstm_92_while_identity_3*backward_lstm_92/while/Identity_3:output:0"O
!backward_lstm_92_while_identity_4*backward_lstm_92/while/Identity_4:output:0"O
!backward_lstm_92_while_identity_5*backward_lstm_92/while/Identity_5:output:0"є
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"ѕ
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ё
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"В
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2r
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:
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
_user_specified_name" backward_lstm_92/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
Г
Д
1__inference_sequential_407_layer_call_fn_55808801
dense_592_input
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCalldense_592_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808783s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_592_input:($
"
_user_specified_name
55808795:($
"
_user_specified_name
55808797
Б
ђ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809017

inputs

states
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
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
Ь
s
G__inference_lambda_96_layer_call_and_return_conditional_losses_55811267
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
й
Ї
#forward_lstm_92_while_cond_55811729<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3>
:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55811729___redundant_placeholder0V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55811729___redundant_placeholder1V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55811729___redundant_placeholder2V
Rforward_lstm_92_while_forward_lstm_92_while_cond_55811729___redundant_placeholder3"
forward_lstm_92_while_identity
б
forward_lstm_92/while/LessLess!forward_lstm_92_while_placeholder:forward_lstm_92_while_less_forward_lstm_92_strided_slice_1*
T0*
_output_shapes
: k
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : :::::Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_92/strided_slice_1:

_output_shapes
:
└
Ы
,__inference_lstm_cell_layer_call_fn_55814124

inputs
states_0
states_1
unknown: 
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2ѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809367o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:($
"
_user_specified_name
55814112:($
"
_user_specified_name
55814114:($
"
_user_specified_name
55814116
▀8
▒
while_body_55812855
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
╠	
═
while_cond_55812997
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55812997___redundant_placeholder06
2while_while_cond_55812997___redundant_placeholder16
2while_while_cond_55812997___redundant_placeholder26
2while_while_cond_55812997___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
╠	
═
while_cond_55809233
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55809233___redundant_placeholder06
2while_while_cond_55809233___redundant_placeholder16
2while_while_cond_55809233___redundant_placeholder26
2while_while_cond_55809233___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
Г
Д
1__inference_sequential_392_layer_call_fn_55808573
dense_577_input
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCalldense_577_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808555s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_577_input:($
"
_user_specified_name
55808567:($
"
_user_specified_name
55808569
У8
▒
while_body_55813284
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        »
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
О
џ
,__inference_dense_593_layer_call_fn_55812582

inputs
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCallМ
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
GPU 2J 8ѓ *P
fKRI
G__inference_dense_593_layer_call_and_return_conditional_losses_55810610f
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
:	ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	ђ
 
_user_specified_nameinputs:($
"
_user_specified_name
55812576:($
"
_user_specified_name
55812578
ЉL
Љ
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813992

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: є
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        т
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:Ы
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55813908*
condR
while_cond_55813907*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
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
╠	
═
while_cond_55809030
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55809030___redundant_placeholder06
2while_while_cond_55809030___redundant_placeholder16
2while_while_cond_55809030___redundant_placeholder26
2while_while_cond_55809030___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
њ
X
,__inference_lambda_97_layer_call_fn_55811273
inputs_0
inputs_1
identityЙ
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_97_layer_call_and_return_conditional_losses_55810246_
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
С
■
G__inference_dense_582_layer_call_and_return_conditional_losses_55808615

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
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
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : ю
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
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Еr
О
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_body_55808268~
ztopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_loop_counterЁ
ђtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_maximum_iterationsF
Btopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholderH
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder_1H
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder_2H
Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder_3}
ytopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_strided_slice_1_0║
хtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0s
atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: u
ctopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: p
btopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: C
?topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identityE
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_1E
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_2E
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_3E
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_4E
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_5{
wtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_strided_slice_1И
│topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorq
_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: s
atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: n
`topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕбWtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpбVtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpбXtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp╣
htopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Њ
Ztopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemхtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0Btopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholderqtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0Э
Vtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpatopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0й
Gtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMulMatMulatopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: Ч
Xtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpctopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0ц
Itopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul_1MatMulDtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder_2`topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: ъ
Dtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/addAddV2Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul:product:0Stopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: Ш
Wtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpbtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Д
Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/BiasAddBiasAddHtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/add:z:0_topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: њ
Ptopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Н
Ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/splitSplitYtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/split/split_dim:output:0Qtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split═
Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/SigmoidSigmoidOtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:¤
Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/Sigmoid_1SigmoidOtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:і
Dtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/mulMulNtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/Sigmoid_1:y:0Dtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:К
Etopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/ReluReluOtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:Ў
Ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/mul_1MulLtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/Sigmoid:y:0Stopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:ј
Ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/add_1AddV2Htopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/mul:z:0Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:¤
Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/Sigmoid_2SigmoidOtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:─
Gtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/Relu_1ReluJtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:Ю
Ftopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/mul_2MulNtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/Sigmoid_2:y:0Utopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:є
[topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemDtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholder_1Btopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholderJtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм~
<topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :№
:topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/addAddV2Btopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_placeholderEtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: ђ
>topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
<topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/add_1AddV2ztopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_loop_counterGtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: В
?topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/IdentityIdentity@topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/add_1:z:0<^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/NoOp*
T0*
_output_shapes
: »
Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_1Identityђtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_maximum_iterations<^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/NoOp*
T0*
_output_shapes
: В
Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_2Identity>topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/add:z:0<^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ў
Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_3Identityktopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0<^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/NoOp*
T0*
_output_shapes
: ђ
Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_4IdentityJtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/mul_2:z:0<^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/NoOp*
T0*
_output_shapes

:ђ
Atopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_5IdentityJtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/add_1:z:0<^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/NoOp*
T0*
_output_shapes

:у
;topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/NoOpNoOpX^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpW^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpY^topk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "І
?topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identityHtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity:output:0"Ј
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_1Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_1:output:0"Ј
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_2Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_2:output:0"Ј
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_3Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_3:output:0"Ј
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_4Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_4:output:0"Ј
Atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_identity_5Jtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/Identity_5:output:0"к
`topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourcebtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"╚
atopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourcectopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"─
_topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_readvariableop_resourceatopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"Ь
│topk_bilstm_moe_bidirectional_92_forward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorхtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_topk_bilstm_moe_bidirectional_92_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0"З
wtopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_strided_slice_1ytopk_bilstm_moe_bidirectional_92_forward_lstm_92_while_topk_bilstm_moe_bidirectional_92_forward_lstm_92_strided_slice_1_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2▓
Wtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpWtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2░
Vtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpVtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2┤
Xtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpXtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:{ w

_output_shapes
: 
]
_user_specified_nameECtopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/loop_counter:Ђ}

_output_shapes
: 
c
_user_specified_nameKItopk_bilstm_moe/bidirectional_92/forward_lstm_92/while/maximum_iterations:
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
_user_specified_nameB@topk_bilstm_moe/bidirectional_92/forward_lstm_92/strided_slice_1:Љї

_output_shapes
: 
r
_user_specified_nameZXtopk_bilstm_moe/bidirectional_92/forward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
ъ
d
H__inference_flatten_92_layer_call_and_return_conditional_losses_55810599

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ђ  T
ReshapeReshapeinputsConst:output:0*
T0*
_output_shapes
:	ђP
IdentityIdentityReshape:output:0*
T0*
_output_shapes
:	ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
Ь
s
G__inference_lambda_98_layer_call_and_return_conditional_losses_55811309
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
а	
╝
2__inference_forward_lstm_92_layer_call_fn_55812785

inputs
unknown: 
	unknown_0: 
	unknown_1: 
identityѕбStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55809651|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'                           
 
_user_specified_nameinputs:($
"
_user_specified_name
55812777:($
"
_user_specified_name
55812779:($
"
_user_specified_name
55812781
у
q
G__inference_lambda_97_layer_call_and_return_conditional_losses_55810246

inputs
inputs_1
identityђ
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
њ
X
,__inference_lambda_98_layer_call_fn_55811297
inputs_0
inputs_1
identity║
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_98_layer_call_and_return_conditional_losses_55810278[
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
┘Y
П
$__inference__traced_restore_55814381
file_prefix3
!assignvariableop_dense_576_kernel:/
!assignvariableop_1_dense_576_bias:6
#assignvariableop_2_dense_593_kernel:	ђ/
!assignvariableop_3_dense_593_bias:5
#assignvariableop_4_dense_577_kernel:/
!assignvariableop_5_dense_577_bias:5
#assignvariableop_6_dense_582_kernel:/
!assignvariableop_7_dense_582_bias:5
#assignvariableop_8_dense_587_kernel:/
!assignvariableop_9_dense_587_bias:6
$assignvariableop_10_dense_592_kernel:0
"assignvariableop_11_dense_592_bias:W
Eassignvariableop_12_bidirectional_92_forward_lstm_92_lstm_cell_kernel: a
Oassignvariableop_13_bidirectional_92_forward_lstm_92_lstm_cell_recurrent_kernel: Q
Cassignvariableop_14_bidirectional_92_forward_lstm_92_lstm_cell_bias: X
Fassignvariableop_15_bidirectional_92_backward_lstm_92_lstm_cell_kernel: b
Passignvariableop_16_bidirectional_92_backward_lstm_92_lstm_cell_recurrent_kernel: R
Dassignvariableop_17_bidirectional_92_backward_lstm_92_lstm_cell_bias: 
identity_19ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Ъ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*┼
value╗BИB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHќ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B §
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOpAssignVariableOp!assignvariableop_dense_576_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_576_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_593_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_593_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_577_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_577_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_582_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_582_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_587_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_587_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dense_592_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dense_592_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:я
AssignVariableOp_12AssignVariableOpEassignvariableop_12_bidirectional_92_forward_lstm_92_lstm_cell_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_13AssignVariableOpOassignvariableop_13_bidirectional_92_forward_lstm_92_lstm_cell_recurrent_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:▄
AssignVariableOp_14AssignVariableOpCassignvariableop_14_bidirectional_92_forward_lstm_92_lstm_cell_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:▀
AssignVariableOp_15AssignVariableOpFassignvariableop_15_bidirectional_92_backward_lstm_92_lstm_cell_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_16AssignVariableOpPassignvariableop_16_bidirectional_92_backward_lstm_92_lstm_cell_recurrent_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_17AssignVariableOpDassignvariableop_17_bidirectional_92_backward_lstm_92_lstm_cell_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 █
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: ц
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
_user_specified_namedense_576/kernel:.*
(
_user_specified_namedense_576/bias:0,
*
_user_specified_namedense_593/kernel:.*
(
_user_specified_namedense_593/bias:0,
*
_user_specified_namedense_577/kernel:.*
(
_user_specified_namedense_577/bias:0,
*
_user_specified_namedense_582/kernel:.*
(
_user_specified_namedense_582/bias:0	,
*
_user_specified_namedense_587/kernel:.
*
(
_user_specified_namedense_587/bias:0,
*
_user_specified_namedense_592/kernel:.*
(
_user_specified_namedense_592/bias:QM
K
_user_specified_name31bidirectional_92/forward_lstm_92/lstm_cell/kernel:[W
U
_user_specified_name=;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel:OK
I
_user_specified_name1/bidirectional_92/forward_lstm_92/lstm_cell/bias:RN
L
_user_specified_name42bidirectional_92/backward_lstm_92/lstm_cell/kernel:\X
V
_user_specified_name><bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel:PL
J
_user_specified_name20bidirectional_92/backward_lstm_92/lstm_cell/bias
У8
▒
while_body_55813908
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        »
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
▒M
Л
$backward_lstm_92_while_body_55811871>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3=
9backward_lstm_92_while_backward_lstm_92_strided_slice_1_0y
ubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_92_while_identity%
!backward_lstm_92_while_identity_1%
!backward_lstm_92_while_identity_2%
!backward_lstm_92_while_identity_3%
!backward_lstm_92_while_identity_4%
!backward_lstm_92_while_identity_5;
7backward_lstm_92_while_backward_lstm_92_strided_slice_1w
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpЎ
Hbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ё
:backward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_92_while_placeholderQbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0И
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0Т
'backward_lstm_92/while/lstm_cell/MatMulMatMulAbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ╝
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0═
)backward_lstm_92/while/lstm_cell/MatMul_1MatMul$backward_lstm_92_while_placeholder_2@backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          К
$backward_lstm_92/while/lstm_cell/addAddV21backward_lstm_92/while/lstm_cell/MatMul:product:03backward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          Х
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0л
(backward_lstm_92/while/lstm_cell/BiasAddBiasAdd(backward_lstm_92/while/lstm_cell/add:z:0?backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
0backward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ў
&backward_lstm_92/while/lstm_cell/splitSplit9backward_lstm_92/while/lstm_cell/split/split_dim:output:01backward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitќ
(backward_lstm_92/while/lstm_cell/SigmoidSigmoid/backward_lstm_92/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         ў
*backward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_92/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         │
$backward_lstm_92/while/lstm_cell/mulMul.backward_lstm_92/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_92_while_placeholder_3*
T0*'
_output_shapes
:         љ
%backward_lstm_92/while/lstm_cell/ReluRelu/backward_lstm_92/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ┬
&backward_lstm_92/while/lstm_cell/mul_1Mul,backward_lstm_92/while/lstm_cell/Sigmoid:y:03backward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         и
&backward_lstm_92/while/lstm_cell/add_1AddV2(backward_lstm_92/while/lstm_cell/mul:z:0*backward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         ў
*backward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_92/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ї
'backward_lstm_92/while/lstm_cell/Relu_1Relu*backward_lstm_92/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         к
&backward_lstm_92/while/lstm_cell/mul_2Mul.backward_lstm_92/while/lstm_cell/Sigmoid_2:y:05backward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         є
;backward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_92_while_placeholder_1"backward_lstm_92_while_placeholder*backward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм^
backward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ј
backward_lstm_92/while/addAddV2"backward_lstm_92_while_placeholder%backward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
backward_lstm_92/while/add_1AddV2:backward_lstm_92_while_backward_lstm_92_while_loop_counter'backward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: ї
backward_lstm_92/while/IdentityIdentity backward_lstm_92/while/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: «
!backward_lstm_92/while/Identity_1Identity@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ї
!backward_lstm_92/while/Identity_2Identitybackward_lstm_92/while/add:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ╣
!backward_lstm_92/while/Identity_3IdentityKbackward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: Е
!backward_lstm_92/while/Identity_4Identity*backward_lstm_92/while/lstm_cell/mul_2:z:0^backward_lstm_92/while/NoOp*
T0*'
_output_shapes
:         Е
!backward_lstm_92/while/Identity_5Identity*backward_lstm_92/while/lstm_cell/add_1:z:0^backward_lstm_92/while/NoOp*
T0*'
_output_shapes
:         у
backward_lstm_92/while/NoOpNoOp8^backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_92_while_backward_lstm_92_strided_slice_19backward_lstm_92_while_backward_lstm_92_strided_slice_1_0"K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0"O
!backward_lstm_92_while_identity_1*backward_lstm_92/while/Identity_1:output:0"O
!backward_lstm_92_while_identity_2*backward_lstm_92/while/Identity_2:output:0"O
!backward_lstm_92_while_identity_3*backward_lstm_92/while/Identity_3:output:0"O
!backward_lstm_92_while_identity_4*backward_lstm_92/while/Identity_4:output:0"O
!backward_lstm_92_while_identity_5*backward_lstm_92/while/Identity_5:output:0"є
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"ѕ
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ё
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"В
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2r
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_92/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
Б
ђ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809220

inputs

states
states_10
matmul_readvariableop_resource: 2
 matmul_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          d
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:          r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0m
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         _
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         X
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
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
њ
X
,__inference_lambda_98_layer_call_fn_55811303
inputs_0
inputs_1
identity║
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_98_layer_call_and_return_conditional_losses_55810676[
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
њ
X
,__inference_lambda_96_layer_call_fn_55811249
inputs_0
inputs_1
identity║
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_96_layer_call_and_return_conditional_losses_55810239[
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
╝:
Є
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55809450

inputs$
lstm_cell_55809368: $
lstm_cell_55809370:  
lstm_cell_55809372: 
identityѕб!lstm_cell/StatefulPartitionedCallбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       т
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskЬ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_55809368lstm_cell_55809370lstm_cell_55809372*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809367n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Џ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_55809368lstm_cell_55809370lstm_cell_55809372*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55809381*
condR
while_cond_55809380*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:($
"
_user_specified_name
55809368:($
"
_user_specified_name
55809370:($
"
_user_specified_name
55809372
њ┐
ъ
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55811671
inputs_0J
8forward_lstm_92_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_92_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource: 
identityѕб1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpб2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбbackward_lstm_92/whileб0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpб1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбforward_lstm_92/while[
forward_lstm_92/ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤m
#forward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
forward_lstm_92/strided_sliceStridedSliceforward_lstm_92/Shape:output:0,forward_lstm_92/strided_slice/stack:output:0.forward_lstm_92/strided_slice/stack_1:output:0.forward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Б
forward_lstm_92/zeros/packedPack&forward_lstm_92/strided_slice:output:0'forward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
forward_lstm_92/zerosFill%forward_lstm_92/zeros/packed:output:0$forward_lstm_92/zeros/Const:output:0*
T0*'
_output_shapes
:         b
 forward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/zeros_1/packedPack&forward_lstm_92/strided_slice:output:0)forward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    б
forward_lstm_92/zeros_1Fill'forward_lstm_92/zeros_1/packed:output:0&forward_lstm_92/zeros_1/Const:output:0*
T0*'
_output_shapes
:         s
forward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          А
forward_lstm_92/transpose	Transposeinputs_0'forward_lstm_92/transpose/perm:output:0*
T0*=
_output_shapes+
):'                           r
forward_lstm_92/Shape_1Shapeforward_lstm_92/transpose:y:0*
T0*
_output_shapes
::ь¤o
%forward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
forward_lstm_92/strided_slice_1StridedSlice forward_lstm_92/Shape_1:output:0.forward_lstm_92/strided_slice_1/stack:output:00forward_lstm_92/strided_slice_1/stack_1:output:00forward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         С
forward_lstm_92/TensorArrayV2TensorListReserve4forward_lstm_92/TensorArrayV2/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмќ
Eforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        љ
7forward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_92/transpose:y:0Nforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмo
%forward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┬
forward_lstm_92/strided_slice_2StridedSliceforward_lstm_92/transpose:y:0.forward_lstm_92/strided_slice_2/stack:output:00forward_lstm_92/strided_slice_2/stack_1:output:00forward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskе
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0┐
 forward_lstm_92/lstm_cell/MatMulMatMul(forward_lstm_92/strided_slice_2:output:07forward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          г
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0╣
"forward_lstm_92/lstm_cell/MatMul_1MatMulforward_lstm_92/zeros:output:09forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ▓
forward_lstm_92/lstm_cell/addAddV2*forward_lstm_92/lstm_cell/MatMul:product:0,forward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          д
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0╗
!forward_lstm_92/lstm_cell/BiasAddBiasAdd!forward_lstm_92/lstm_cell/add:z:08forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          k
)forward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ё
forward_lstm_92/lstm_cell/splitSplit2forward_lstm_92/lstm_cell/split/split_dim:output:0*forward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitѕ
!forward_lstm_92/lstm_cell/SigmoidSigmoid(forward_lstm_92/lstm_cell/split:output:0*
T0*'
_output_shapes
:         і
#forward_lstm_92/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_92/lstm_cell/split:output:1*
T0*'
_output_shapes
:         А
forward_lstm_92/lstm_cell/mulMul'forward_lstm_92/lstm_cell/Sigmoid_1:y:0 forward_lstm_92/zeros_1:output:0*
T0*'
_output_shapes
:         ѓ
forward_lstm_92/lstm_cell/ReluRelu(forward_lstm_92/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Г
forward_lstm_92/lstm_cell/mul_1Mul%forward_lstm_92/lstm_cell/Sigmoid:y:0,forward_lstm_92/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         б
forward_lstm_92/lstm_cell/add_1AddV2!forward_lstm_92/lstm_cell/mul:z:0#forward_lstm_92/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         і
#forward_lstm_92/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_92/lstm_cell/split:output:3*
T0*'
_output_shapes
:         
 forward_lstm_92/lstm_cell/Relu_1Relu#forward_lstm_92/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         ▒
forward_lstm_92/lstm_cell/mul_2Mul'forward_lstm_92/lstm_cell/Sigmoid_2:y:0.forward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ~
-forward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       У
forward_lstm_92/TensorArrayV2_1TensorListReserve6forward_lstm_92/TensorArrayV2_1/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмV
forward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         d
"forward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : └
forward_lstm_92/whileWhile+forward_lstm_92/while/loop_counter:output:01forward_lstm_92/while/maximum_iterations:output:0forward_lstm_92/time:output:0(forward_lstm_92/TensorArrayV2_1:handle:0forward_lstm_92/zeros:output:0 forward_lstm_92/zeros_1:output:0(forward_lstm_92/strided_slice_1:output:0Gforward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_92_lstm_cell_matmul_readvariableop_resource:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*/
body'R%
#forward_lstm_92_while_body_55811442*/
cond'R%
#forward_lstm_92_while_cond_55811441*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Љ
@forward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ч
2forward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_92/while:output:3Iforward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0x
%forward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         q
'forward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
forward_lstm_92/strided_slice_3StridedSlice;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_92/strided_slice_3/stack:output:00forward_lstm_92/strided_slice_3/stack_1:output:00forward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_masku
 forward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¤
forward_lstm_92/transpose_1	Transpose;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_92/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  k
forward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
backward_lstm_92/ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤n
$backward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
backward_lstm_92/strided_sliceStridedSlicebackward_lstm_92/Shape:output:0-backward_lstm_92/strided_slice/stack:output:0/backward_lstm_92/strided_slice/stack_1:output:0/backward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :д
backward_lstm_92/zeros/packedPack'backward_lstm_92/strided_slice:output:0(backward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ъ
backward_lstm_92/zerosFill&backward_lstm_92/zeros/packed:output:0%backward_lstm_92/zeros/Const:output:0*
T0*'
_output_shapes
:         c
!backward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ф
backward_lstm_92/zeros_1/packedPack'backward_lstm_92/strided_slice:output:0*backward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ц
backward_lstm_92/zeros_1Fill(backward_lstm_92/zeros_1/packed:output:0'backward_lstm_92/zeros_1/Const:output:0*
T0*'
_output_shapes
:         t
backward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Б
backward_lstm_92/transpose	Transposeinputs_0(backward_lstm_92/transpose/perm:output:0*
T0*=
_output_shapes+
):'                           t
backward_lstm_92/Shape_1Shapebackward_lstm_92/transpose:y:0*
T0*
_output_shapes
::ь¤p
&backward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 backward_lstm_92/strided_slice_1StridedSlice!backward_lstm_92/Shape_1:output:0/backward_lstm_92/strided_slice_1/stack:output:01backward_lstm_92/strided_slice_1/stack_1:output:01backward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         у
backward_lstm_92/TensorArrayV2TensorListReserve5backward_lstm_92/TensorArrayV2/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмi
backward_lstm_92/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ╣
backward_lstm_92/ReverseV2	ReverseV2backward_lstm_92/transpose:y:0(backward_lstm_92/ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           Ќ
Fbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ў
8backward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_92/ReverseV2:output:0Obackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмp
&backward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
 backward_lstm_92/strided_slice_2StridedSlicebackward_lstm_92/transpose:y:0/backward_lstm_92/strided_slice_2/stack:output:01backward_lstm_92/strided_slice_2/stack_1:output:01backward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskф
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0┬
!backward_lstm_92/lstm_cell/MatMulMatMul)backward_lstm_92/strided_slice_2:output:08backward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          «
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0╝
#backward_lstm_92/lstm_cell/MatMul_1MatMulbackward_lstm_92/zeros:output:0:backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          х
backward_lstm_92/lstm_cell/addAddV2+backward_lstm_92/lstm_cell/MatMul:product:0-backward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          е
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Й
"backward_lstm_92/lstm_cell/BiasAddBiasAdd"backward_lstm_92/lstm_cell/add:z:09backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          l
*backward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Є
 backward_lstm_92/lstm_cell/splitSplit3backward_lstm_92/lstm_cell/split/split_dim:output:0+backward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitі
"backward_lstm_92/lstm_cell/SigmoidSigmoid)backward_lstm_92/lstm_cell/split:output:0*
T0*'
_output_shapes
:         ї
$backward_lstm_92/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_92/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ц
backward_lstm_92/lstm_cell/mulMul(backward_lstm_92/lstm_cell/Sigmoid_1:y:0!backward_lstm_92/zeros_1:output:0*
T0*'
_output_shapes
:         ё
backward_lstm_92/lstm_cell/ReluRelu)backward_lstm_92/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ░
 backward_lstm_92/lstm_cell/mul_1Mul&backward_lstm_92/lstm_cell/Sigmoid:y:0-backward_lstm_92/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         Ц
 backward_lstm_92/lstm_cell/add_1AddV2"backward_lstm_92/lstm_cell/mul:z:0$backward_lstm_92/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         ї
$backward_lstm_92/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_92/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ђ
!backward_lstm_92/lstm_cell/Relu_1Relu$backward_lstm_92/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         ┤
 backward_lstm_92/lstm_cell/mul_2Mul(backward_lstm_92/lstm_cell/Sigmoid_2:y:0/backward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         
.backward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       в
 backward_lstm_92/TensorArrayV2_1TensorListReserve7backward_lstm_92/TensorArrayV2_1/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмW
backward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         e
#backward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ╬
backward_lstm_92/whileWhile,backward_lstm_92/while/loop_counter:output:02backward_lstm_92/while/maximum_iterations:output:0backward_lstm_92/time:output:0)backward_lstm_92/TensorArrayV2_1:handle:0backward_lstm_92/zeros:output:0!backward_lstm_92/zeros_1:output:0)backward_lstm_92/strided_slice_1:output:0Hbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_92_lstm_cell_matmul_readvariableop_resource;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*0
body(R&
$backward_lstm_92_while_body_55811583*0
cond(R&
$backward_lstm_92_while_cond_55811582*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations њ
Abackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ■
3backward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_92/while:output:3Jbackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0y
&backward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         r
(backward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▄
 backward_lstm_92/strided_slice_3StridedSlice<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_92/strided_slice_3/stack:output:01backward_lstm_92/strided_slice_3/stack_1:output:01backward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskv
!backward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          м
backward_lstm_92/transpose_1	Transpose<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_92/transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  l
backward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    X
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:љ
	ReverseV2	ReverseV2 backward_lstm_92/transpose_1:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ц
concatConcatV2forward_lstm_92/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  k
IdentityIdentityconcat:output:0^NoOp*
T0*4
_output_shapes"
 :                  ѕ
NoOpNoOp2^backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_92/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_92/while1^forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_92/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:'                           : : : : : : 2f
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_92/whilebackward_lstm_92/while2d
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_92/whileforward_lstm_92/while:g c
=
_output_shapes+
):'                           
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
к

љ
3__inference_bidirectional_92_layer_call_fn_55811383

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallњ
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
GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55810965j
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
55811369:($
"
_user_specified_name
55811371:($
"
_user_specified_name
55811373:($
"
_user_specified_name
55811375:($
"
_user_specified_name
55811377:($
"
_user_specified_name
55811379
к

љ
3__inference_bidirectional_92_layer_call_fn_55811366

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallњ
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
GPU 2J 8ѓ *W
fRRP
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55810567j
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
55811352:($
"
_user_specified_name
55811354:($
"
_user_specified_name
55811356:($
"
_user_specified_name
55811358:($
"
_user_specified_name
55811360:($
"
_user_specified_name
55811362
вK
Њ
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813557
inputs_0:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileK
ShapeShapeinputs_0*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: }
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*4
_output_shapes"
 :                  є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       т
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55813473*
condR
while_cond_55813472*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
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
Д	
Я
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808546
dense_577_input$
dense_577_55808540: 
dense_577_55808542:
identityѕб!dense_577/StatefulPartitionedCallЄ
!dense_577/StatefulPartitionedCallStatefulPartitionedCalldense_577_inputdense_577_55808540dense_577_55808542*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_577_layer_call_and_return_conditional_losses_55808539}
IdentityIdentity*dense_577/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         F
NoOpNoOp"^dense_577/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 2F
!dense_577/StatefulPartitionedCall!dense_577/StatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_577_input:($
"
_user_specified_name
55808540:($
"
_user_specified_name
55808542
ќ	
┐
3__inference_backward_lstm_92_layer_call_fn_55813379
inputs_0
unknown: 
	unknown_0: 
	unknown_1: 
identityѕбStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55809303|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0:($
"
_user_specified_name
55813371:($
"
_user_specified_name
55813373:($
"
_user_specified_name
55813375
і

g
H__inference_dropout_92_layer_call_and_return_conditional_losses_55810592

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?_
dropout/MulMulinputsdropout/Const:output:0*
T0*"
_output_shapes
:b
dropout/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         Є
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
 *═╠L>А
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*"
_output_shapes
:T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ј
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
д
■
G__inference_dense_576_layer_call_and_return_conditional_losses_55810221

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ђ     p
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	ђѓ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђd
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
И
А
$backward_lstm_92_while_cond_55810478>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3@
<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55810478___redundant_placeholder0X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55810478___redundant_placeholder1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55810478___redundant_placeholder2X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55810478___redundant_placeholder3#
backward_lstm_92_while_identity
д
backward_lstm_92/while/LessLess"backward_lstm_92_while_placeholder<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_92/while/IdentityIdentitybackward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.: : : : ::: :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:
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
_user_specified_name" backward_lstm_92/strided_slice_1:

_output_shapes
:
▒M
Л
$backward_lstm_92_while_body_55811583>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3=
9backward_lstm_92_while_backward_lstm_92_strided_slice_1_0y
ubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_92_while_identity%
!backward_lstm_92_while_identity_1%
!backward_lstm_92_while_identity_2%
!backward_lstm_92_while_identity_3%
!backward_lstm_92_while_identity_4%
!backward_lstm_92_while_identity_5;
7backward_lstm_92_while_backward_lstm_92_strided_slice_1w
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpЎ
Hbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        ё
:backward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_92_while_placeholderQbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0И
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0Т
'backward_lstm_92/while/lstm_cell/MatMulMatMulAbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ╝
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0═
)backward_lstm_92/while/lstm_cell/MatMul_1MatMul$backward_lstm_92_while_placeholder_2@backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          К
$backward_lstm_92/while/lstm_cell/addAddV21backward_lstm_92/while/lstm_cell/MatMul:product:03backward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          Х
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0л
(backward_lstm_92/while/lstm_cell/BiasAddBiasAdd(backward_lstm_92/while/lstm_cell/add:z:0?backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          r
0backward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ў
&backward_lstm_92/while/lstm_cell/splitSplit9backward_lstm_92/while/lstm_cell/split/split_dim:output:01backward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitќ
(backward_lstm_92/while/lstm_cell/SigmoidSigmoid/backward_lstm_92/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         ў
*backward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_92/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         │
$backward_lstm_92/while/lstm_cell/mulMul.backward_lstm_92/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_92_while_placeholder_3*
T0*'
_output_shapes
:         љ
%backward_lstm_92/while/lstm_cell/ReluRelu/backward_lstm_92/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ┬
&backward_lstm_92/while/lstm_cell/mul_1Mul,backward_lstm_92/while/lstm_cell/Sigmoid:y:03backward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         и
&backward_lstm_92/while/lstm_cell/add_1AddV2(backward_lstm_92/while/lstm_cell/mul:z:0*backward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         ў
*backward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_92/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         Ї
'backward_lstm_92/while/lstm_cell/Relu_1Relu*backward_lstm_92/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         к
&backward_lstm_92/while/lstm_cell/mul_2Mul.backward_lstm_92/while/lstm_cell/Sigmoid_2:y:05backward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         є
;backward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_92_while_placeholder_1"backward_lstm_92_while_placeholder*backward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм^
backward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ј
backward_lstm_92/while/addAddV2"backward_lstm_92_while_placeholder%backward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
backward_lstm_92/while/add_1AddV2:backward_lstm_92_while_backward_lstm_92_while_loop_counter'backward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: ї
backward_lstm_92/while/IdentityIdentity backward_lstm_92/while/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: «
!backward_lstm_92/while/Identity_1Identity@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ї
!backward_lstm_92/while/Identity_2Identitybackward_lstm_92/while/add:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ╣
!backward_lstm_92/while/Identity_3IdentityKbackward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: Е
!backward_lstm_92/while/Identity_4Identity*backward_lstm_92/while/lstm_cell/mul_2:z:0^backward_lstm_92/while/NoOp*
T0*'
_output_shapes
:         Е
!backward_lstm_92/while/Identity_5Identity*backward_lstm_92/while/lstm_cell/add_1:z:0^backward_lstm_92/while/NoOp*
T0*'
_output_shapes
:         у
backward_lstm_92/while/NoOpNoOp8^backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_92_while_backward_lstm_92_strided_slice_19backward_lstm_92_while_backward_lstm_92_strided_slice_1_0"K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0"O
!backward_lstm_92_while_identity_1*backward_lstm_92/while/Identity_1:output:0"O
!backward_lstm_92_while_identity_2*backward_lstm_92/while/Identity_2:output:0"O
!backward_lstm_92_while_identity_3*backward_lstm_92/while/Identity_3:output:0"O
!backward_lstm_92_while_identity_4*backward_lstm_92/while/Identity_4:output:0"O
!backward_lstm_92_while_identity_5*backward_lstm_92/while/Identity_5:output:0"є
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"ѕ
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ё
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"В
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2r
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_92/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
Й
▄
2__inference_topk_bilstm_moe_layer_call_fn_55811033
input_3
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

unknown_15:	ђ

unknown_16:
identityѕбStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU 2J 8ѓ *V
fQRO
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810617f
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
_user_specified_name	input_3:($
"
_user_specified_name
55810995:($
"
_user_specified_name
55810997:($
"
_user_specified_name
55810999:($
"
_user_specified_name
55811001:($
"
_user_specified_name
55811003:($
"
_user_specified_name
55811005:($
"
_user_specified_name
55811007:($
"
_user_specified_name
55811009:(	$
"
_user_specified_name
55811011:(
$
"
_user_specified_name
55811013:($
"
_user_specified_name
55811015:($
"
_user_specified_name
55811017:($
"
_user_specified_name
55811019:($
"
_user_specified_name
55811021:($
"
_user_specified_name
55811023:($
"
_user_specified_name
55811025:($
"
_user_specified_name
55811027:($
"
_user_specified_name
55811029
Д	
Я
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808555
dense_577_input$
dense_577_55808549: 
dense_577_55808551:
identityѕб!dense_577/StatefulPartitionedCallЄ
!dense_577/StatefulPartitionedCallStatefulPartitionedCalldense_577_inputdense_577_55808549dense_577_55808551*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_577_layer_call_and_return_conditional_losses_55808539}
IdentityIdentity*dense_577/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         F
NoOpNoOp"^dense_577/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 2F
!dense_577/StatefulPartitionedCall!dense_577/StatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_577_input:($
"
_user_specified_name
55808549:($
"
_user_specified_name
55808551
▄
А
$backward_lstm_92_while_cond_55811582>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3@
<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55811582___redundant_placeholder0X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55811582___redundant_placeholder1X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55811582___redundant_placeholder2X
Tbackward_lstm_92_while_backward_lstm_92_while_cond_55811582___redundant_placeholder3#
backward_lstm_92_while_identity
д
backward_lstm_92/while/LessLess"backward_lstm_92_while_placeholder<backward_lstm_92_while_less_backward_lstm_92_strided_slice_1*
T0*
_output_shapes
: m
backward_lstm_92/while/IdentityIdentitybackward_lstm_92/while/Less:z:0*
T0
*
_output_shapes
: "K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : :::::[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :XT

_output_shapes
: 
:
_user_specified_name" backward_lstm_92/strided_slice_1:

_output_shapes
:
ЌL
▒
#forward_lstm_92_while_body_55811730<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3;
7forward_lstm_92_while_forward_lstm_92_strided_slice_1_0w
sforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_92_while_identity$
 forward_lstm_92_while_identity_1$
 forward_lstm_92_while_identity_2$
 forward_lstm_92_while_identity_3$
 forward_lstm_92_while_identity_4$
 forward_lstm_92_while_identity_59
5forward_lstm_92_while_forward_lstm_92_strided_slice_1u
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpў
Gforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"         
9forward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_92_while_placeholderPforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0Х
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0с
&forward_lstm_92/while/lstm_cell/MatMulMatMul@forward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ║
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0╩
(forward_lstm_92/while/lstm_cell/MatMul_1MatMul#forward_lstm_92_while_placeholder_2?forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ─
#forward_lstm_92/while/lstm_cell/addAddV20forward_lstm_92/while/lstm_cell/MatMul:product:02forward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ┤
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0═
'forward_lstm_92/while/lstm_cell/BiasAddBiasAdd'forward_lstm_92/while/lstm_cell/add:z:0>forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          q
/forward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ќ
%forward_lstm_92/while/lstm_cell/splitSplit8forward_lstm_92/while/lstm_cell/split/split_dim:output:00forward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitћ
'forward_lstm_92/while/lstm_cell/SigmoidSigmoid.forward_lstm_92/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:         ќ
)forward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_92/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ░
#forward_lstm_92/while/lstm_cell/mulMul-forward_lstm_92/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_92_while_placeholder_3*
T0*'
_output_shapes
:         ј
$forward_lstm_92/while/lstm_cell/ReluRelu.forward_lstm_92/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:         ┐
%forward_lstm_92/while/lstm_cell/mul_1Mul+forward_lstm_92/while/lstm_cell/Sigmoid:y:02forward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ┤
%forward_lstm_92/while/lstm_cell/add_1AddV2'forward_lstm_92/while/lstm_cell/mul:z:0)forward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         ќ
)forward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_92/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:         І
&forward_lstm_92/while/lstm_cell/Relu_1Relu)forward_lstm_92/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         ├
%forward_lstm_92/while/lstm_cell/mul_2Mul-forward_lstm_92/while/lstm_cell/Sigmoid_2:y:04forward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ѓ
:forward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_92_while_placeholder_1!forward_lstm_92_while_placeholder)forward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм]
forward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ї
forward_lstm_92/while/addAddV2!forward_lstm_92_while_placeholder$forward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/while/add_1AddV28forward_lstm_92_while_forward_lstm_92_while_loop_counter&forward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: Ѕ
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: ф
 forward_lstm_92/while/Identity_1Identity>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ѕ
 forward_lstm_92/while/Identity_2Identityforward_lstm_92/while/add:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Х
 forward_lstm_92/while/Identity_3IdentityJforward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: д
 forward_lstm_92/while/Identity_4Identity)forward_lstm_92/while/lstm_cell/mul_2:z:0^forward_lstm_92/while/NoOp*
T0*'
_output_shapes
:         д
 forward_lstm_92/while/Identity_5Identity)forward_lstm_92/while/lstm_cell/add_1:z:0^forward_lstm_92/while/NoOp*
T0*'
_output_shapes
:         с
forward_lstm_92/while/NoOpNoOp7^forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_92_while_forward_lstm_92_strided_slice_17forward_lstm_92_while_forward_lstm_92_strided_slice_1_0"I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0"M
 forward_lstm_92_while_identity_1)forward_lstm_92/while/Identity_1:output:0"M
 forward_lstm_92_while_identity_2)forward_lstm_92/while/Identity_2:output:0"M
 forward_lstm_92_while_identity_3)forward_lstm_92/while/Identity_3:output:0"M
 forward_lstm_92_while_identity_4)forward_lstm_92/while/Identity_4:output:0"M
 forward_lstm_92_while_identity_5)forward_lstm_92/while/Identity_5:output:0"ё
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"є
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ѓ
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"У
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2p
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :WS

_output_shapes
: 
9
_user_specified_name!forward_lstm_92/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
лK
Л
$backward_lstm_92_while_body_55810479>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3=
9backward_lstm_92_while_backward_lstm_92_strided_slice_1_0y
ubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_92_while_identity%
!backward_lstm_92_while_identity_1%
!backward_lstm_92_while_identity_2%
!backward_lstm_92_while_identity_3%
!backward_lstm_92_while_identity_4%
!backward_lstm_92_while_identity_5;
7backward_lstm_92_while_backward_lstm_92_strided_slice_1w
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpЎ
Hbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ы
:backward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_92_while_placeholderQbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0И
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0П
'backward_lstm_92/while/lstm_cell/MatMulMatMulAbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ╝
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0─
)backward_lstm_92/while/lstm_cell/MatMul_1MatMul$backward_lstm_92_while_placeholder_2@backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: Й
$backward_lstm_92/while/lstm_cell/addAddV21backward_lstm_92/while/lstm_cell/MatMul:product:03backward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: Х
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0К
(backward_lstm_92/while/lstm_cell/BiasAddBiasAdd(backward_lstm_92/while/lstm_cell/add:z:0?backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: r
0backward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
&backward_lstm_92/while/lstm_cell/splitSplit9backward_lstm_92/while/lstm_cell/split/split_dim:output:01backward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitЇ
(backward_lstm_92/while/lstm_cell/SigmoidSigmoid/backward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:Ј
*backward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:ф
$backward_lstm_92/while/lstm_cell/mulMul.backward_lstm_92/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:Є
%backward_lstm_92/while/lstm_cell/ReluRelu/backward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:╣
&backward_lstm_92/while/lstm_cell/mul_1Mul,backward_lstm_92/while/lstm_cell/Sigmoid:y:03backward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:«
&backward_lstm_92/while/lstm_cell/add_1AddV2(backward_lstm_92/while/lstm_cell/mul:z:0*backward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ј
*backward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:ё
'backward_lstm_92/while/lstm_cell/Relu_1Relu*backward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:й
&backward_lstm_92/while/lstm_cell/mul_2Mul.backward_lstm_92/while/lstm_cell/Sigmoid_2:y:05backward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:є
;backward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_92_while_placeholder_1"backward_lstm_92_while_placeholder*backward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм^
backward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ј
backward_lstm_92/while/addAddV2"backward_lstm_92_while_placeholder%backward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
backward_lstm_92/while/add_1AddV2:backward_lstm_92_while_backward_lstm_92_while_loop_counter'backward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: ї
backward_lstm_92/while/IdentityIdentity backward_lstm_92/while/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: «
!backward_lstm_92/while/Identity_1Identity@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ї
!backward_lstm_92/while/Identity_2Identitybackward_lstm_92/while/add:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ╣
!backward_lstm_92/while/Identity_3IdentityKbackward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: а
!backward_lstm_92/while/Identity_4Identity*backward_lstm_92/while/lstm_cell/mul_2:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes

:а
!backward_lstm_92/while/Identity_5Identity*backward_lstm_92/while/lstm_cell/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes

:у
backward_lstm_92/while/NoOpNoOp8^backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_92_while_backward_lstm_92_strided_slice_19backward_lstm_92_while_backward_lstm_92_strided_slice_1_0"K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0"O
!backward_lstm_92_while_identity_1*backward_lstm_92/while/Identity_1:output:0"O
!backward_lstm_92_while_identity_2*backward_lstm_92/while/Identity_2:output:0"O
!backward_lstm_92_while_identity_3*backward_lstm_92/while/Identity_3:output:0"O
!backward_lstm_92_while_identity_4*backward_lstm_92/while/Identity_4:output:0"O
!backward_lstm_92_while_identity_5*backward_lstm_92/while/Identity_5:output:0"є
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"ѕ
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ё
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"В
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2r
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:
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
_user_specified_name" backward_lstm_92/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
└
Ы
,__inference_lstm_cell_layer_call_fn_55814107

inputs
states_0
states_1
unknown: 
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2ѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809220o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:($
"
_user_specified_name
55814095:($
"
_user_specified_name
55814097:($
"
_user_specified_name
55814099
ЉL
Љ
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55810113

inputs:
(lstm_cell_matmul_readvariableop_resource: <
*lstm_cell_matmul_1_readvariableop_resource: 7
)lstm_cell_biasadd_readvariableop_resource: 
identityѕб lstm_cell/BiasAdd/ReadVariableOpбlstm_cell/MatMul/ReadVariableOpб!lstm_cell/MatMul_1/ReadVariableOpбwhileI
ShapeShapeinputs*
T0*
_output_shapes
::ь¤]
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
valueB:Л
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
:         R
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
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
	transpose	Transposeinputstranspose/perm:output:0*
T0*=
_output_shapes+
):'                           R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::ь¤_
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
valueB:█
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
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмX
ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: є
	ReverseV2	ReverseV2transpose:y:0ReverseV2/axis:output:0*
T0*=
_output_shapes+
):'                           є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        т
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensorReverseV2:output:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
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
valueB:Ы
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*0
_output_shapes
:                  *
shrink_axis_maskѕ
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Ј
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ї
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0Ѕ
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ѓ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          є
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0І
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          [
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :н
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splith
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:         q
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         b
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:         }
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         r
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         j
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:         _
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
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
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55810029*
condR
while_cond_55810028*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  Њ
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:'                           : : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:e a
=
_output_shapes+
):'                           
 
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
Т
q
G__inference_lambda_98_layer_call_and_return_conditional_losses_55810278

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
С$
о
while_body_55809234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_lstm_cell_55809258_0: ,
while_lstm_cell_55809260_0: (
while_lstm_cell_55809262_0: 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_lstm_cell_55809258: *
while_lstm_cell_55809260: &
while_lstm_cell_55809262: ѕб'while/lstm_cell/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0г
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_55809258_0while_lstm_cell_55809260_0while_lstm_cell_55809262_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55809220┘
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: Ї
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         Ї
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_55809258while_lstm_cell_55809258_0"6
while_lstm_cell_55809260while_lstm_cell_55809260_0"6
while_lstm_cell_55809262while_lstm_cell_55809262_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2R
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
:         :-)
'
_output_shapes
:         :GC
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
55809258:(	$
"
_user_specified_name
55809260:(
$
"
_user_specified_name
55809262
Г
Д
1__inference_sequential_397_layer_call_fn_55808640
dense_582_input
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCalldense_582_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808622s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_582_input:($
"
_user_specified_name
55808634:($
"
_user_specified_name
55808636
ХJ
▒
#forward_lstm_92_while_body_55810736<
8forward_lstm_92_while_forward_lstm_92_while_loop_counterB
>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations%
!forward_lstm_92_while_placeholder'
#forward_lstm_92_while_placeholder_1'
#forward_lstm_92_while_placeholder_2'
#forward_lstm_92_while_placeholder_3;
7forward_lstm_92_while_forward_lstm_92_strided_slice_1_0w
sforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0R
@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: T
Bforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: O
Aforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: "
forward_lstm_92_while_identity$
 forward_lstm_92_while_identity_1$
 forward_lstm_92_while_identity_2$
 forward_lstm_92_while_identity_3$
 forward_lstm_92_while_identity_4$
 forward_lstm_92_while_identity_59
5forward_lstm_92_while_forward_lstm_92_strided_slice_1u
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorP
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: R
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: M
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpў
Gforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ь
9forward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0!forward_lstm_92_while_placeholderPforward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0Х
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0┌
&forward_lstm_92/while/lstm_cell/MatMulMatMul@forward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0=forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ║
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0┴
(forward_lstm_92/while/lstm_cell/MatMul_1MatMul#forward_lstm_92_while_placeholder_2?forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: ╗
#forward_lstm_92/while/lstm_cell/addAddV20forward_lstm_92/while/lstm_cell/MatMul:product:02forward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: ┤
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0─
'forward_lstm_92/while/lstm_cell/BiasAddBiasAdd'forward_lstm_92/while/lstm_cell/add:z:0>forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: q
/forward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ы
%forward_lstm_92/while/lstm_cell/splitSplit8forward_lstm_92/while/lstm_cell/split/split_dim:output:00forward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitІ
'forward_lstm_92/while/lstm_cell/SigmoidSigmoid.forward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:Ї
)forward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid.forward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:Д
#forward_lstm_92/while/lstm_cell/mulMul-forward_lstm_92/while/lstm_cell/Sigmoid_1:y:0#forward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:Ё
$forward_lstm_92/while/lstm_cell/ReluRelu.forward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:Х
%forward_lstm_92/while/lstm_cell/mul_1Mul+forward_lstm_92/while/lstm_cell/Sigmoid:y:02forward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Ф
%forward_lstm_92/while/lstm_cell/add_1AddV2'forward_lstm_92/while/lstm_cell/mul:z:0)forward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ї
)forward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid.forward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:ѓ
&forward_lstm_92/while/lstm_cell/Relu_1Relu)forward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:║
%forward_lstm_92/while/lstm_cell/mul_2Mul-forward_lstm_92/while/lstm_cell/Sigmoid_2:y:04forward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:ѓ
:forward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#forward_lstm_92_while_placeholder_1!forward_lstm_92_while_placeholder)forward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм]
forward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ї
forward_lstm_92/while/addAddV2!forward_lstm_92_while_placeholder$forward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: _
forward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/while/add_1AddV28forward_lstm_92_while_forward_lstm_92_while_loop_counter&forward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: Ѕ
forward_lstm_92/while/IdentityIdentityforward_lstm_92/while/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: ф
 forward_lstm_92/while/Identity_1Identity>forward_lstm_92_while_forward_lstm_92_while_maximum_iterations^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ѕ
 forward_lstm_92/while/Identity_2Identityforward_lstm_92/while/add:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Х
 forward_lstm_92/while/Identity_3IdentityJforward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes
: Ю
 forward_lstm_92/while/Identity_4Identity)forward_lstm_92/while/lstm_cell/mul_2:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes

:Ю
 forward_lstm_92/while/Identity_5Identity)forward_lstm_92/while/lstm_cell/add_1:z:0^forward_lstm_92/while/NoOp*
T0*
_output_shapes

:с
forward_lstm_92/while/NoOpNoOp7^forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6^forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp8^forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "p
5forward_lstm_92_while_forward_lstm_92_strided_slice_17forward_lstm_92_while_forward_lstm_92_strided_slice_1_0"I
forward_lstm_92_while_identity'forward_lstm_92/while/Identity:output:0"M
 forward_lstm_92_while_identity_1)forward_lstm_92/while/Identity_1:output:0"M
 forward_lstm_92_while_identity_2)forward_lstm_92/while/Identity_2:output:0"M
 forward_lstm_92_while_identity_3)forward_lstm_92/while/Identity_3:output:0"M
 forward_lstm_92_while_identity_4)forward_lstm_92/while/Identity_4:output:0"M
 forward_lstm_92_while_identity_5)forward_lstm_92/while/Identity_5:output:0"ё
?forward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceAforward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"є
@forward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceBforward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ѓ
>forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource@forward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"У
qforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensorsforward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_forward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2p
6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp6forward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2n
5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp5forward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2r
7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp7forward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:Z V

_output_shapes
: 
<
_user_specified_name$"forward_lstm_92/while/loop_counter:`\

_output_shapes
: 
B
_user_specified_name*(forward_lstm_92/while/maximum_iterations:
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
_user_specified_name!forward_lstm_92/strided_slice_1:ok

_output_shapes
: 
Q
_user_specified_name97forward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
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
Г
Д
1__inference_sequential_392_layer_call_fn_55808564
dense_577_input
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCalldense_577_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808546s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_577_input:($
"
_user_specified_name
55808558:($
"
_user_specified_name
55808560
╠	
═
while_cond_55809380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55809380___redundant_placeholder06
2while_while_cond_55809380___redundant_placeholder16
2while_while_cond_55809380___redundant_placeholder26
2while_while_cond_55809380___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
С
Ў
,__inference_dense_576_layer_call_fn_55811228

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallО
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
GPU 2J 8ѓ *P
fKRI
G__inference_dense_576_layer_call_and_return_conditional_losses_55810221j
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
55811222:($
"
_user_specified_name
55811224
Т
q
G__inference_lambda_96_layer_call_and_return_conditional_losses_55810239

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
й╣
ю
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55812535

inputsJ
8forward_lstm_92_lstm_cell_matmul_readvariableop_resource: L
:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: G
9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource: K
9backward_lstm_92_lstm_cell_matmul_readvariableop_resource: M
;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource: H
:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource: 
identityѕб1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpб2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбbackward_lstm_92/whileб0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpб/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpб1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpбforward_lstm_92/whilej
forward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         m
#forward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%forward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%forward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
forward_lstm_92/strided_sliceStridedSliceforward_lstm_92/Shape:output:0,forward_lstm_92/strided_slice/stack:output:0.forward_lstm_92/strided_slice/stack_1:output:0.forward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
forward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Б
forward_lstm_92/zeros/packedPack&forward_lstm_92/strided_slice:output:0'forward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:`
forward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Њ
forward_lstm_92/zerosFill%forward_lstm_92/zeros/packed:output:0$forward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:b
 forward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Д
forward_lstm_92/zeros_1/packedPack&forward_lstm_92/strided_slice:output:0)forward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:b
forward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ў
forward_lstm_92/zeros_1Fill'forward_lstm_92/zeros_1/packed:output:0&forward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:s
forward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ё
forward_lstm_92/transpose	Transposeinputs'forward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:l
forward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         o
%forward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
forward_lstm_92/strided_slice_1StridedSlice forward_lstm_92/Shape_1:output:0.forward_lstm_92/strided_slice_1/stack:output:00forward_lstm_92/strided_slice_1/stack_1:output:00forward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+forward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         С
forward_lstm_92/TensorArrayV2TensorListReserve4forward_lstm_92/TensorArrayV2/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмќ
Eforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      љ
7forward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorforward_lstm_92/transpose:y:0Nforward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмo
%forward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'forward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
forward_lstm_92/strided_slice_2StridedSliceforward_lstm_92/transpose:y:0.forward_lstm_92/strided_slice_2/stack:output:00forward_lstm_92/strided_slice_2/stack_1:output:00forward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskе
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp8forward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Х
 forward_lstm_92/lstm_cell/MatMulMatMul(forward_lstm_92/strided_slice_2:output:07forward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: г
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0░
"forward_lstm_92/lstm_cell/MatMul_1MatMulforward_lstm_92/zeros:output:09forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: Е
forward_lstm_92/lstm_cell/addAddV2*forward_lstm_92/lstm_cell/MatMul:product:0,forward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: д
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0▓
!forward_lstm_92/lstm_cell/BiasAddBiasAdd!forward_lstm_92/lstm_cell/add:z:08forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: k
)forward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Я
forward_lstm_92/lstm_cell/splitSplit2forward_lstm_92/lstm_cell/split/split_dim:output:0*forward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_split
!forward_lstm_92/lstm_cell/SigmoidSigmoid(forward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:Ђ
#forward_lstm_92/lstm_cell/Sigmoid_1Sigmoid(forward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:ў
forward_lstm_92/lstm_cell/mulMul'forward_lstm_92/lstm_cell/Sigmoid_1:y:0 forward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:y
forward_lstm_92/lstm_cell/ReluRelu(forward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:ц
forward_lstm_92/lstm_cell/mul_1Mul%forward_lstm_92/lstm_cell/Sigmoid:y:0,forward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:Ў
forward_lstm_92/lstm_cell/add_1AddV2!forward_lstm_92/lstm_cell/mul:z:0#forward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ђ
#forward_lstm_92/lstm_cell/Sigmoid_2Sigmoid(forward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:v
 forward_lstm_92/lstm_cell/Relu_1Relu#forward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:е
forward_lstm_92/lstm_cell/mul_2Mul'forward_lstm_92/lstm_cell/Sigmoid_2:y:0.forward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:~
-forward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      У
forward_lstm_92/TensorArrayV2_1TensorListReserve6forward_lstm_92/TensorArrayV2_1/element_shape:output:0(forward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмV
forward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : s
(forward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         d
"forward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ю
forward_lstm_92/whileWhile+forward_lstm_92/while/loop_counter:output:01forward_lstm_92/while/maximum_iterations:output:0forward_lstm_92/time:output:0(forward_lstm_92/TensorArrayV2_1:handle:0forward_lstm_92/zeros:output:0 forward_lstm_92/zeros_1:output:0(forward_lstm_92/strided_slice_1:output:0Gforward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:08forward_lstm_92_lstm_cell_matmul_readvariableop_resource:forward_lstm_92_lstm_cell_matmul_1_readvariableop_resource9forward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
#forward_lstm_92_while_body_55812306*/
cond'R%
#forward_lstm_92_while_cond_55812305*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations Љ
@forward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ж
2forward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackforward_lstm_92/while:output:3Iforward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0x
%forward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         q
'forward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'forward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
forward_lstm_92/strided_slice_3StridedSlice;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0.forward_lstm_92/strided_slice_3/stack:output:00forward_lstm_92/strided_slice_3/stack_1:output:00forward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_masku
 forward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
forward_lstm_92/transpose_1	Transpose;forward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0)forward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:k
forward_lstm_92/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
backward_lstm_92/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"         n
$backward_lstm_92/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&backward_lstm_92/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&backward_lstm_92/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
backward_lstm_92/strided_sliceStridedSlicebackward_lstm_92/Shape:output:0-backward_lstm_92/strided_slice/stack:output:0/backward_lstm_92/strided_slice/stack_1:output:0/backward_lstm_92/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
backward_lstm_92/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :д
backward_lstm_92/zeros/packedPack'backward_lstm_92/strided_slice:output:0(backward_lstm_92/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:a
backward_lstm_92/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ќ
backward_lstm_92/zerosFill&backward_lstm_92/zeros/packed:output:0%backward_lstm_92/zeros/Const:output:0*
T0*
_output_shapes

:c
!backward_lstm_92/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ф
backward_lstm_92/zeros_1/packedPack'backward_lstm_92/strided_slice:output:0*backward_lstm_92/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:c
backward_lstm_92/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ю
backward_lstm_92/zeros_1Fill(backward_lstm_92/zeros_1/packed:output:0'backward_lstm_92/zeros_1/Const:output:0*
T0*
_output_shapes

:t
backward_lstm_92/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          є
backward_lstm_92/transpose	Transposeinputs(backward_lstm_92/transpose/perm:output:0*
T0*"
_output_shapes
:m
backward_lstm_92/Shape_1Const*
_output_shapes
:*
dtype0*!
valueB"         p
&backward_lstm_92/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 backward_lstm_92/strided_slice_1StridedSlice!backward_lstm_92/Shape_1:output:0/backward_lstm_92/strided_slice_1/stack:output:01backward_lstm_92/strided_slice_1/stack_1:output:01backward_lstm_92/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
,backward_lstm_92/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         у
backward_lstm_92/TensorArrayV2TensorListReserve5backward_lstm_92/TensorArrayV2/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмi
backward_lstm_92/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ъ
backward_lstm_92/ReverseV2	ReverseV2backward_lstm_92/transpose:y:0(backward_lstm_92/ReverseV2/axis:output:0*
T0*"
_output_shapes
:Ќ
Fbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      ў
8backward_lstm_92/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#backward_lstm_92/ReverseV2:output:0Obackward_lstm_92/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмp
&backward_lstm_92/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(backward_lstm_92/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:х
 backward_lstm_92/strided_slice_2StridedSlicebackward_lstm_92/transpose:y:0/backward_lstm_92/strided_slice_2/stack:output:01backward_lstm_92/strided_slice_2/stack_1:output:01backward_lstm_92/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskф
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOpReadVariableOp9backward_lstm_92_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype0╣
!backward_lstm_92/lstm_cell/MatMulMatMul)backward_lstm_92/strided_slice_2:output:08backward_lstm_92/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: «
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype0│
#backward_lstm_92/lstm_cell/MatMul_1MatMulbackward_lstm_92/zeros:output:0:backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: г
backward_lstm_92/lstm_cell/addAddV2+backward_lstm_92/lstm_cell/MatMul:product:0-backward_lstm_92/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: е
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0х
"backward_lstm_92/lstm_cell/BiasAddBiasAdd"backward_lstm_92/lstm_cell/add:z:09backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: l
*backward_lstm_92/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :с
 backward_lstm_92/lstm_cell/splitSplit3backward_lstm_92/lstm_cell/split/split_dim:output:0+backward_lstm_92/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitЂ
"backward_lstm_92/lstm_cell/SigmoidSigmoid)backward_lstm_92/lstm_cell/split:output:0*
T0*
_output_shapes

:Ѓ
$backward_lstm_92/lstm_cell/Sigmoid_1Sigmoid)backward_lstm_92/lstm_cell/split:output:1*
T0*
_output_shapes

:Џ
backward_lstm_92/lstm_cell/mulMul(backward_lstm_92/lstm_cell/Sigmoid_1:y:0!backward_lstm_92/zeros_1:output:0*
T0*
_output_shapes

:{
backward_lstm_92/lstm_cell/ReluRelu)backward_lstm_92/lstm_cell/split:output:2*
T0*
_output_shapes

:Д
 backward_lstm_92/lstm_cell/mul_1Mul&backward_lstm_92/lstm_cell/Sigmoid:y:0-backward_lstm_92/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:ю
 backward_lstm_92/lstm_cell/add_1AddV2"backward_lstm_92/lstm_cell/mul:z:0$backward_lstm_92/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ѓ
$backward_lstm_92/lstm_cell/Sigmoid_2Sigmoid)backward_lstm_92/lstm_cell/split:output:3*
T0*
_output_shapes

:x
!backward_lstm_92/lstm_cell/Relu_1Relu$backward_lstm_92/lstm_cell/add_1:z:0*
T0*
_output_shapes

:Ф
 backward_lstm_92/lstm_cell/mul_2Mul(backward_lstm_92/lstm_cell/Sigmoid_2:y:0/backward_lstm_92/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:
.backward_lstm_92/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      в
 backward_lstm_92/TensorArrayV2_1TensorListReserve7backward_lstm_92/TensorArrayV2_1/element_shape:output:0)backward_lstm_92/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмW
backward_lstm_92/timeConst*
_output_shapes
: *
dtype0*
value	B : t
)backward_lstm_92/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         e
#backward_lstm_92/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ф
backward_lstm_92/whileWhile,backward_lstm_92/while/loop_counter:output:02backward_lstm_92/while/maximum_iterations:output:0backward_lstm_92/time:output:0)backward_lstm_92/TensorArrayV2_1:handle:0backward_lstm_92/zeros:output:0!backward_lstm_92/zeros_1:output:0)backward_lstm_92/strided_slice_1:output:0Hbackward_lstm_92/TensorArrayUnstack/TensorListFromTensor:output_handle:09backward_lstm_92_lstm_cell_matmul_readvariableop_resource;backward_lstm_92_lstm_cell_matmul_1_readvariableop_resource:backward_lstm_92_lstm_cell_biasadd_readvariableop_resource*
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
$backward_lstm_92_while_body_55812447*0
cond(R&
$backward_lstm_92_while_cond_55812446*9
output_shapes(
&: : : : ::: : : : : *
parallel_iterations њ
Abackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      В
3backward_lstm_92/TensorArrayV2Stack/TensorListStackTensorListStackbackward_lstm_92/while:output:3Jbackward_lstm_92/TensorArrayV2Stack/TensorListStack/element_shape:output:0*"
_output_shapes
:*
element_dtype0y
&backward_lstm_92/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         r
(backward_lstm_92/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(backward_lstm_92/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:М
 backward_lstm_92/strided_slice_3StridedSlice<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0/backward_lstm_92/strided_slice_3/stack:output:01backward_lstm_92/strided_slice_3/stack_1:output:01backward_lstm_92/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_maskv
!backward_lstm_92/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          └
backward_lstm_92/transpose_1	Transpose<backward_lstm_92/TensorArrayV2Stack/TensorListStack:tensor:0*backward_lstm_92/transpose_1/perm:output:0*
T0*"
_output_shapes
:l
backward_lstm_92/runtimeConst"/device:CPU:0*
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
	ReverseV2	ReverseV2 backward_lstm_92/transpose_1:y:0ReverseV2/axis:output:0*
T0*"
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Њ
concatConcatV2forward_lstm_92/transpose_1:y:0ReverseV2:output:0concat/axis:output:0*
N*
T0*"
_output_shapes
:Y
IdentityIdentityconcat:output:0^NoOp*
T0*"
_output_shapes
:ѕ
NoOpNoOp2^backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1^backward_lstm_92/lstm_cell/MatMul/ReadVariableOp3^backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^backward_lstm_92/while1^forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0^forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2^forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp^forward_lstm_92/while*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2f
1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp1backward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2d
0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp0backward_lstm_92/lstm_cell/MatMul/ReadVariableOp2h
2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2backward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp20
backward_lstm_92/whilebackward_lstm_92/while2d
0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp0forward_lstm_92/lstm_cell/BiasAdd/ReadVariableOp2b
/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp/forward_lstm_92/lstm_cell/MatMul/ReadVariableOp2f
1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp1forward_lstm_92/lstm_cell/MatMul_1/ReadVariableOp2.
forward_lstm_92/whileforward_lstm_92/while:J F
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
Ь
s
G__inference_lambda_98_layer_call_and_return_conditional_losses_55811315
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
Д	
Я
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808698
dense_587_input$
dense_587_55808692: 
dense_587_55808694:
identityѕб!dense_587/StatefulPartitionedCallЄ
!dense_587/StatefulPartitionedCallStatefulPartitionedCalldense_587_inputdense_587_55808692dense_587_55808694*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_dense_587_layer_call_and_return_conditional_losses_55808691}
IdentityIdentity*dense_587/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         F
NoOpNoOp"^dense_587/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 2F
!dense_587/StatefulPartitionedCall!dense_587/StatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_587_input:($
"
_user_specified_name
55808692:($
"
_user_specified_name
55808694
њ
X
,__inference_lambda_96_layer_call_fn_55811255
inputs_0
inputs_1
identity║
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
GPU 2J 8ѓ *P
fKRI
G__inference_lambda_96_layer_call_and_return_conditional_losses_55810637[
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
╠	
═
while_cond_55813472
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_55813472___redundant_placeholder06
2while_while_cond_55813472___redundant_placeholder16
2while_while_cond_55813472___redundant_placeholder26
2while_while_cond_55813472___redundant_placeholder3
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
@: : : : :         :         : :::::J F
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
:         :-)
'
_output_shapes
:         :GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
д
■
G__inference_dense_576_layer_call_and_return_conditional_losses_55811243

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0h
Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ђ     p
Tensordot/ReshapeReshapeinputs Tensordot/Reshape/shape:output:0*
T0*
_output_shapes
:	ђѓ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	ђd
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
У8
▒
while_body_55813141
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
/while_lstm_cell_biasadd_readvariableop_resource: ѕб&while/lstm_cell/BiasAdd/ReadVariableOpб%while/lstm_cell/MatMul/ReadVariableOpб'while/lstm_cell/MatMul_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"        »
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*0
_output_shapes
:                  *
element_dtype0ќ
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0│
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          џ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0џ
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:          ћ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:          ћ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0Ю
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Т
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         :         :         :         *
	num_splitt
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:         ђ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         n
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:         Ј
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:         ё
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         v
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:         k
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:         ┬
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУмM
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
: є
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: v
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         v
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         Б

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
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2P
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
:         :-)
'
_output_shapes
:         :GC
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
└
Ы
,__inference_lstm_cell_layer_call_fn_55814009

inputs
states_0
states_1
unknown: 
	unknown_0: 
	unknown_1: 
identity

identity_1

identity_2ѕбStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55808872o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states_0:QM
'
_output_shapes
:         
"
_user_specified_name
states_1:($
"
_user_specified_name
55813997:($
"
_user_specified_name
55813999:($
"
_user_specified_name
55814001
С
■
G__inference_dense_577_layer_call_and_return_conditional_losses_55812632

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
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
::ь¤Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : ю
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
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:         e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:         V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Г
Д
1__inference_sequential_402_layer_call_fn_55808716
dense_587_input
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCalldense_587_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *U
fPRN
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808698s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
+
_output_shapes
:         
)
_user_specified_namedense_587_input:($
"
_user_specified_name
55808710:($
"
_user_specified_name
55808712
лK
Л
$backward_lstm_92_while_body_55812159>
:backward_lstm_92_while_backward_lstm_92_while_loop_counterD
@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations&
"backward_lstm_92_while_placeholder(
$backward_lstm_92_while_placeholder_1(
$backward_lstm_92_while_placeholder_2(
$backward_lstm_92_while_placeholder_3=
9backward_lstm_92_while_backward_lstm_92_strided_slice_1_0y
ubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0S
Abackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0: U
Cbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0: P
Bbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0: #
backward_lstm_92_while_identity%
!backward_lstm_92_while_identity_1%
!backward_lstm_92_while_identity_2%
!backward_lstm_92_while_identity_3%
!backward_lstm_92_while_identity_4%
!backward_lstm_92_while_identity_5;
7backward_lstm_92_while_backward_lstm_92_strided_slice_1w
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorQ
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resource: S
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource: N
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource: ѕб7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpб6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpб8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpЎ
Hbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      Ы
:backward_lstm_92/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0"backward_lstm_92_while_placeholderQbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype0И
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype0П
'backward_lstm_92/while/lstm_cell/MatMulMatMulAbackward_lstm_92/while/TensorArrayV2Read/TensorListGetItem:item:0>backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

: ╝
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype0─
)backward_lstm_92/while/lstm_cell/MatMul_1MatMul$backward_lstm_92_while_placeholder_2@backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

: Й
$backward_lstm_92/while/lstm_cell/addAddV21backward_lstm_92/while/lstm_cell/MatMul:product:03backward_lstm_92/while/lstm_cell/MatMul_1:product:0*
T0*
_output_shapes

: Х
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype0К
(backward_lstm_92/while/lstm_cell/BiasAddBiasAdd(backward_lstm_92/while/lstm_cell/add:z:0?backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

: r
0backward_lstm_92/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ш
&backward_lstm_92/while/lstm_cell/splitSplit9backward_lstm_92/while/lstm_cell/split/split_dim:output:01backward_lstm_92/while/lstm_cell/BiasAdd:output:0*
T0*<
_output_shapes*
(::::*
	num_splitЇ
(backward_lstm_92/while/lstm_cell/SigmoidSigmoid/backward_lstm_92/while/lstm_cell/split:output:0*
T0*
_output_shapes

:Ј
*backward_lstm_92/while/lstm_cell/Sigmoid_1Sigmoid/backward_lstm_92/while/lstm_cell/split:output:1*
T0*
_output_shapes

:ф
$backward_lstm_92/while/lstm_cell/mulMul.backward_lstm_92/while/lstm_cell/Sigmoid_1:y:0$backward_lstm_92_while_placeholder_3*
T0*
_output_shapes

:Є
%backward_lstm_92/while/lstm_cell/ReluRelu/backward_lstm_92/while/lstm_cell/split:output:2*
T0*
_output_shapes

:╣
&backward_lstm_92/while/lstm_cell/mul_1Mul,backward_lstm_92/while/lstm_cell/Sigmoid:y:03backward_lstm_92/while/lstm_cell/Relu:activations:0*
T0*
_output_shapes

:«
&backward_lstm_92/while/lstm_cell/add_1AddV2(backward_lstm_92/while/lstm_cell/mul:z:0*backward_lstm_92/while/lstm_cell/mul_1:z:0*
T0*
_output_shapes

:Ј
*backward_lstm_92/while/lstm_cell/Sigmoid_2Sigmoid/backward_lstm_92/while/lstm_cell/split:output:3*
T0*
_output_shapes

:ё
'backward_lstm_92/while/lstm_cell/Relu_1Relu*backward_lstm_92/while/lstm_cell/add_1:z:0*
T0*
_output_shapes

:й
&backward_lstm_92/while/lstm_cell/mul_2Mul.backward_lstm_92/while/lstm_cell/Sigmoid_2:y:05backward_lstm_92/while/lstm_cell/Relu_1:activations:0*
T0*
_output_shapes

:є
;backward_lstm_92/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem$backward_lstm_92_while_placeholder_1"backward_lstm_92_while_placeholder*backward_lstm_92/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:жУм^
backward_lstm_92/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ј
backward_lstm_92/while/addAddV2"backward_lstm_92_while_placeholder%backward_lstm_92/while/add/y:output:0*
T0*
_output_shapes
: `
backward_lstm_92/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ф
backward_lstm_92/while/add_1AddV2:backward_lstm_92_while_backward_lstm_92_while_loop_counter'backward_lstm_92/while/add_1/y:output:0*
T0*
_output_shapes
: ї
backward_lstm_92/while/IdentityIdentity backward_lstm_92/while/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: «
!backward_lstm_92/while/Identity_1Identity@backward_lstm_92_while_backward_lstm_92_while_maximum_iterations^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ї
!backward_lstm_92/while/Identity_2Identitybackward_lstm_92/while/add:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: ╣
!backward_lstm_92/while/Identity_3IdentityKbackward_lstm_92/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes
: а
!backward_lstm_92/while/Identity_4Identity*backward_lstm_92/while/lstm_cell/mul_2:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes

:а
!backward_lstm_92/while/Identity_5Identity*backward_lstm_92/while/lstm_cell/add_1:z:0^backward_lstm_92/while/NoOp*
T0*
_output_shapes

:у
backward_lstm_92/while/NoOpNoOp8^backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7^backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp9^backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp*
_output_shapes
 "t
7backward_lstm_92_while_backward_lstm_92_strided_slice_19backward_lstm_92_while_backward_lstm_92_strided_slice_1_0"K
backward_lstm_92_while_identity(backward_lstm_92/while/Identity:output:0"O
!backward_lstm_92_while_identity_1*backward_lstm_92/while/Identity_1:output:0"O
!backward_lstm_92_while_identity_2*backward_lstm_92/while/Identity_2:output:0"O
!backward_lstm_92_while_identity_3*backward_lstm_92/while/Identity_3:output:0"O
!backward_lstm_92_while_identity_4*backward_lstm_92/while/Identity_4:output:0"O
!backward_lstm_92_while_identity_5*backward_lstm_92/while/Identity_5:output:0"є
@backward_lstm_92_while_lstm_cell_biasadd_readvariableop_resourceBbackward_lstm_92_while_lstm_cell_biasadd_readvariableop_resource_0"ѕ
Abackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resourceCbackward_lstm_92_while_lstm_cell_matmul_1_readvariableop_resource_0"ё
?backward_lstm_92_while_lstm_cell_matmul_readvariableop_resourceAbackward_lstm_92_while_lstm_cell_matmul_readvariableop_resource_0"В
sbackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensorubackward_lstm_92_while_tensorarrayv2read_tensorlistgetitem_backward_lstm_92_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : ::: : : : : 2r
7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp7backward_lstm_92/while/lstm_cell/BiasAdd/ReadVariableOp2p
6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp6backward_lstm_92/while/lstm_cell/MatMul/ReadVariableOp2t
8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp8backward_lstm_92/while/lstm_cell/MatMul_1/ReadVariableOp:[ W

_output_shapes
: 
=
_user_specified_name%#backward_lstm_92/while/loop_counter:a]

_output_shapes
: 
C
_user_specified_name+)backward_lstm_92/while/maximum_iterations:
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
_user_specified_name" backward_lstm_92/strided_slice_1:pl

_output_shapes
: 
R
_user_specified_name:8backward_lstm_92/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource"╩L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ъ
serving_defaultі
6
input_3+
serving_default_input_3:04
	dense_593'
StatefulPartitionedCall:0tensorflow/serving/predict:гО
Њ
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
Я
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
╩
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
#/_self_saveable_object_factories"
_tf_keras_layer
╩
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
Ш
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
Ш
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
Ш
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
Ш
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
╩
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
#a_self_saveable_object_factories"
_tf_keras_layer
ы
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
р
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
q_random_generator
#r_self_saveable_object_factories"
_tf_keras_layer
╩
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
#y_self_saveable_object_factories"
_tf_keras_layer
с
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses
ђkernel
	Ђbias
$ѓ_self_saveable_object_factories"
_tf_keras_layer
Х
"0
#1
Ѓ2
ё3
Ё4
є5
Є6
ѕ7
Ѕ8
і9
І10
ї11
Ї12
ј13
Ј14
љ15
ђ16
Ђ17"
trackable_list_wrapper
Х
"0
#1
Ѓ2
ё3
Ё4
є5
Є6
ѕ7
Ѕ8
і9
І10
ї11
Ї12
ј13
Ј14
љ15
ђ16
Ђ17"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
Љnon_trainable_variables
њlayers
Њmetrics
 ћlayer_regularization_losses
Ћlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
█
ќtrace_0
Ќtrace_12а
2__inference_topk_bilstm_moe_layer_call_fn_55811033
2__inference_topk_bilstm_moe_layer_call_fn_55811074х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zќtrace_0zЌtrace_1
Љ
ўtrace_0
Ўtrace_12о
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810617
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810992х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zўtrace_0zЎtrace_1
╬B╦
#__inference__wrapped_model_55808506input_3"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
-
џserving_default"
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
▓
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
У
аtrace_02╔
,__inference_dense_576_layer_call_fn_55811228ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zаtrace_0
Ѓ
Аtrace_02С
G__inference_dense_576_layer_call_and_return_conditional_losses_55811243ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zАtrace_0
": 2dense_576/kernel
:2dense_576/bias
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
▓
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
¤
Дtrace_0
еtrace_12ћ
,__inference_lambda_96_layer_call_fn_55811249
,__inference_lambda_96_layer_call_fn_55811255х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zДtrace_0zеtrace_1
Ё
Еtrace_0
фtrace_12╩
G__inference_lambda_96_layer_call_and_return_conditional_losses_55811261
G__inference_lambda_96_layer_call_and_return_conditional_losses_55811267х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЕtrace_0zфtrace_1
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Фnon_trainable_variables
гlayers
Гmetrics
 «layer_regularization_losses
»layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
¤
░trace_0
▒trace_12ћ
,__inference_lambda_97_layer_call_fn_55811273
,__inference_lambda_97_layer_call_fn_55811279х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z░trace_0z▒trace_1
Ё
▓trace_0
│trace_12╩
G__inference_lambda_97_layer_call_and_return_conditional_losses_55811285
G__inference_lambda_97_layer_call_and_return_conditional_losses_55811291х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▓trace_0z│trace_1
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
ж
┤	variables
хtrainable_variables
Хregularization_losses
и	keras_api
И__call__
+╣&call_and_return_all_conditional_losses
Ѓkernel
	ёbias
$║_self_saveable_object_factories"
_tf_keras_layer
0
Ѓ0
ё1"
trackable_list_wrapper
0
Ѓ0
ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╗non_trainable_variables
╝layers
йmetrics
 Йlayer_regularization_losses
┐layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
┘
└trace_0
┴trace_12ъ
1__inference_sequential_392_layer_call_fn_55808564
1__inference_sequential_392_layer_call_fn_55808573х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z└trace_0z┴trace_1
Ј
┬trace_0
├trace_12н
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808546
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808555х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┬trace_0z├trace_1
 "
trackable_dict_wrapper
ж
─	variables
┼trainable_variables
кregularization_losses
К	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses
Ёkernel
	єbias
$╩_self_saveable_object_factories"
_tf_keras_layer
0
Ё0
є1"
trackable_list_wrapper
0
Ё0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
┘
лtrace_0
Лtrace_12ъ
1__inference_sequential_397_layer_call_fn_55808640
1__inference_sequential_397_layer_call_fn_55808649х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zлtrace_0zЛtrace_1
Ј
мtrace_0
Мtrace_12н
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808622
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808631х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zмtrace_0zМtrace_1
 "
trackable_dict_wrapper
ж
н	variables
Нtrainable_variables
оregularization_losses
О	keras_api
п__call__
+┘&call_and_return_all_conditional_losses
Єkernel
	ѕbias
$┌_self_saveable_object_factories"
_tf_keras_layer
0
Є0
ѕ1"
trackable_list_wrapper
0
Є0
ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
█non_trainable_variables
▄layers
Пmetrics
 яlayer_regularization_losses
▀layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
┘
Яtrace_0
рtrace_12ъ
1__inference_sequential_402_layer_call_fn_55808716
1__inference_sequential_402_layer_call_fn_55808725х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЯtrace_0zрtrace_1
Ј
Рtrace_0
сtrace_12н
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808698
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808707х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zРtrace_0zсtrace_1
 "
trackable_dict_wrapper
ж
С	variables
тtrainable_variables
Тregularization_losses
у	keras_api
У__call__
+ж&call_and_return_all_conditional_losses
Ѕkernel
	іbias
$Ж_self_saveable_object_factories"
_tf_keras_layer
0
Ѕ0
і1"
trackable_list_wrapper
0
Ѕ0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
┘
­trace_0
ыtrace_12ъ
1__inference_sequential_407_layer_call_fn_55808792
1__inference_sequential_407_layer_call_fn_55808801х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z­trace_0zыtrace_1
Ј
Ыtrace_0
зtrace_12н
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808774
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808783х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЫtrace_0zзtrace_1
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
▓
Зnon_trainable_variables
шlayers
Шmetrics
 эlayer_regularization_losses
Эlayer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
¤
щtrace_0
Щtrace_12ћ
,__inference_lambda_98_layer_call_fn_55811297
,__inference_lambda_98_layer_call_fn_55811303х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zщtrace_0zЩtrace_1
Ё
чtrace_0
Чtrace_12╩
G__inference_lambda_98_layer_call_and_return_conditional_losses_55811309
G__inference_lambda_98_layer_call_and_return_conditional_losses_55811315х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zчtrace_0zЧtrace_1
 "
trackable_dict_wrapper
P
І0
ї1
Ї2
ј3
Ј4
љ5"
trackable_list_wrapper
P
І0
ї1
Ї2
ј3
Ј4
љ5"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
§non_trainable_variables
■layers
 metrics
 ђlayer_regularization_losses
Ђlayer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
Ц
ѓtrace_0
Ѓtrace_1
ёtrace_2
Ёtrace_32▓
3__inference_bidirectional_92_layer_call_fn_55811332
3__inference_bidirectional_92_layer_call_fn_55811349
3__inference_bidirectional_92_layer_call_fn_55811366
3__inference_bidirectional_92_layer_call_fn_55811383█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѓtrace_0zЃtrace_1zёtrace_2zЁtrace_3
Љ
єtrace_0
Єtrace_1
ѕtrace_2
Ѕtrace_32ъ
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55811671
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55811959
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55812247
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55812535█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zєtrace_0zЄtrace_1zѕtrace_2zЅtrace_3
Ѕ
і	variables
Іtrainable_variables
їregularization_losses
Ї	keras_api
ј__call__
+Ј&call_and_return_all_conditional_losses
љ_random_generator
	Љcell
њ
state_spec
$Њ_self_saveable_object_factories"
_tf_keras_rnn_layer
Ѕ
ћ	variables
Ћtrainable_variables
ќregularization_losses
Ќ	keras_api
ў__call__
+Ў&call_and_return_all_conditional_losses
џ_random_generator
	Џcell
ю
state_spec
$Ю_self_saveable_object_factories"
_tf_keras_rnn_layer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ъnon_trainable_variables
Ъlayers
аmetrics
 Аlayer_regularization_losses
бlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
┼
Бtrace_0
цtrace_12і
-__inference_dropout_92_layer_call_fn_55812540
-__inference_dropout_92_layer_call_fn_55812545Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zБtrace_0zцtrace_1
ч
Цtrace_0
дtrace_12└
H__inference_dropout_92_layer_call_and_return_conditional_losses_55812557
H__inference_dropout_92_layer_call_and_return_conditional_losses_55812562Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЦtrace_0zдtrace_1
D
$Д_self_saveable_object_factories"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
еnon_trainable_variables
Еlayers
фmetrics
 Фlayer_regularization_losses
гlayer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
ж
Гtrace_02╩
-__inference_flatten_92_layer_call_fn_55812567ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zГtrace_0
ё
«trace_02т
H__inference_flatten_92_layer_call_and_return_conditional_losses_55812573ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z«trace_0
 "
trackable_dict_wrapper
0
ђ0
Ђ1"
trackable_list_wrapper
0
ђ0
Ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
»non_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
У
┤trace_02╔
,__inference_dense_593_layer_call_fn_55812582ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┤trace_0
Ѓ
хtrace_02С
G__inference_dense_593_layer_call_and_return_conditional_losses_55812592ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zхtrace_0
#:!	ђ2dense_593/kernel
:2dense_593/bias
 "
trackable_dict_wrapper
": 2dense_577/kernel
:2dense_577/bias
": 2dense_582/kernel
:2dense_582/bias
": 2dense_587/kernel
:2dense_587/bias
": 2dense_592/kernel
:2dense_592/bias
C:A 21bidirectional_92/forward_lstm_92/lstm_cell/kernel
M:K 2;bidirectional_92/forward_lstm_92/lstm_cell/recurrent_kernel
=:; 2/bidirectional_92/forward_lstm_92/lstm_cell/bias
D:B 22bidirectional_92/backward_lstm_92/lstm_cell/kernel
N:L 2<bidirectional_92/backward_lstm_92/lstm_cell/recurrent_kernel
>:< 20bidirectional_92/backward_lstm_92/lstm_cell/bias
 "
trackable_list_wrapper
ъ
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
ЩBэ
2__inference_topk_bilstm_moe_layer_call_fn_55811033input_3"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
2__inference_topk_bilstm_moe_layer_call_fn_55811074input_3"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810617input_3"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810992input_3"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
═B╩
&__inference_signature_wrapper_55811219input_3"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
оBМ
,__inference_dense_576_layer_call_fn_55811228inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ыBЬ
G__inference_dense_576_layer_call_and_return_conditional_losses_55811243inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
 BЧ
,__inference_lambda_96_layer_call_fn_55811249inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
,__inference_lambda_96_layer_call_fn_55811255inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
G__inference_lambda_96_layer_call_and_return_conditional_losses_55811261inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
G__inference_lambda_96_layer_call_and_return_conditional_losses_55811267inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
 BЧ
,__inference_lambda_97_layer_call_fn_55811273inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
,__inference_lambda_97_layer_call_fn_55811279inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
G__inference_lambda_97_layer_call_and_return_conditional_losses_55811285inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
G__inference_lambda_97_layer_call_and_return_conditional_losses_55811291inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
Ѓ0
ё1"
trackable_list_wrapper
0
Ѓ0
ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Хnon_trainable_variables
иlayers
Иmetrics
 ╣layer_regularization_losses
║layer_metrics
┤	variables
хtrainable_variables
Хregularization_losses
И__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
У
╗trace_02╔
,__inference_dense_577_layer_call_fn_55812601ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╗trace_0
Ѓ
╝trace_02С
G__inference_dense_577_layer_call_and_return_conditional_losses_55812632ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╝trace_0
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
ЂB■
1__inference_sequential_392_layer_call_fn_55808564dense_577_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
1__inference_sequential_392_layer_call_fn_55808573dense_577_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808546dense_577_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808555dense_577_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
Ё0
є1"
trackable_list_wrapper
0
Ё0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
йnon_trainable_variables
Йlayers
┐metrics
 └layer_regularization_losses
┴layer_metrics
─	variables
┼trainable_variables
кregularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
У
┬trace_02╔
,__inference_dense_582_layer_call_fn_55812641ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┬trace_0
Ѓ
├trace_02С
G__inference_dense_582_layer_call_and_return_conditional_losses_55812672ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z├trace_0
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
ЂB■
1__inference_sequential_397_layer_call_fn_55808640dense_582_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
1__inference_sequential_397_layer_call_fn_55808649dense_582_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808622dense_582_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808631dense_582_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
Є0
ѕ1"
trackable_list_wrapper
0
Є0
ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
─non_trainable_variables
┼layers
кmetrics
 Кlayer_regularization_losses
╚layer_metrics
н	variables
Нtrainable_variables
оregularization_losses
п__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
У
╔trace_02╔
,__inference_dense_587_layer_call_fn_55812681ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╔trace_0
Ѓ
╩trace_02С
G__inference_dense_587_layer_call_and_return_conditional_losses_55812712ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╩trace_0
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
ЂB■
1__inference_sequential_402_layer_call_fn_55808716dense_587_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
1__inference_sequential_402_layer_call_fn_55808725dense_587_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808698dense_587_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808707dense_587_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
Ѕ0
і1"
trackable_list_wrapper
0
Ѕ0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
╦non_trainable_variables
╠layers
═metrics
 ╬layer_regularization_losses
¤layer_metrics
С	variables
тtrainable_variables
Тregularization_losses
У__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
У
лtrace_02╔
,__inference_dense_592_layer_call_fn_55812721ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zлtrace_0
Ѓ
Лtrace_02С
G__inference_dense_592_layer_call_and_return_conditional_losses_55812752ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЛtrace_0
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
ЂB■
1__inference_sequential_407_layer_call_fn_55808792dense_592_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЂB■
1__inference_sequential_407_layer_call_fn_55808801dense_592_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808774dense_592_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
юBЎ
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808783dense_592_input"х
«▓ф
FullArgSpec)
args!џ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
 BЧ
,__inference_lambda_98_layer_call_fn_55811297inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
,__inference_lambda_98_layer_call_fn_55811303inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
G__inference_lambda_98_layer_call_and_return_conditional_losses_55811309inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
џBЌ
G__inference_lambda_98_layer_call_and_return_conditional_losses_55811315inputs_0inputs_1"х
«▓ф
FullArgSpec)
args!џ
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsб

 
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
бBЪ
3__inference_bidirectional_92_layer_call_fn_55811332inputs_0"█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
бBЪ
3__inference_bidirectional_92_layer_call_fn_55811349inputs_0"█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
аBЮ
3__inference_bidirectional_92_layer_call_fn_55811366inputs"█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
аBЮ
3__inference_bidirectional_92_layer_call_fn_55811383inputs"█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
йB║
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55811671inputs_0"█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
йB║
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55811959inputs_0"█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╗BИ
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55812247inputs"█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╗BИ
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55812535inputs"█
н▓л
FullArgSpecG
args?џ<
jinputs

jtraining
jmask
jinitial_state
j	constants
varargs
 
varkw
 
defaultsб
p 

 

 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
8
І0
ї1
Ї2"
trackable_list_wrapper
8
І0
ї1
Ї2"
trackable_list_wrapper
 "
trackable_list_wrapper
┼
мstates
Мnon_trainable_variables
нlayers
Нmetrics
 оlayer_regularization_losses
Оlayer_metrics
і	variables
Іtrainable_variables
їregularization_losses
ј__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
љ
пtrace_0
┘trace_1
┌trace_2
█trace_32Ю
2__inference_forward_lstm_92_layer_call_fn_55812763
2__inference_forward_lstm_92_layer_call_fn_55812774
2__inference_forward_lstm_92_layer_call_fn_55812785
2__inference_forward_lstm_92_layer_call_fn_55812796╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zпtrace_0z┘trace_1z┌trace_2z█trace_3
Ч
▄trace_0
Пtrace_1
яtrace_2
▀trace_32Ѕ
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55812939
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813082
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813225
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813368╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▄trace_0zПtrace_1zяtrace_2z▀trace_3
D
$Я_self_saveable_object_factories"
_generic_user_object
Е
р	variables
Рtrainable_variables
сregularization_losses
С	keras_api
т__call__
+Т&call_and_return_all_conditional_losses
у_random_generator
У
state_size
Іkernel
їrecurrent_kernel
	Їbias
$ж_self_saveable_object_factories"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
ј0
Ј1
љ2"
trackable_list_wrapper
8
ј0
Ј1
љ2"
trackable_list_wrapper
 "
trackable_list_wrapper
┼
Жstates
вnon_trainable_variables
Вlayers
ьmetrics
 Ьlayer_regularization_losses
№layer_metrics
ћ	variables
Ћtrainable_variables
ќregularization_losses
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
ћ
­trace_0
ыtrace_1
Ыtrace_2
зtrace_32А
3__inference_backward_lstm_92_layer_call_fn_55813379
3__inference_backward_lstm_92_layer_call_fn_55813390
3__inference_backward_lstm_92_layer_call_fn_55813401
3__inference_backward_lstm_92_layer_call_fn_55813412╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z­trace_0zыtrace_1zЫtrace_2zзtrace_3
ђ
Зtrace_0
шtrace_1
Шtrace_2
эtrace_32Ї
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813557
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813702
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813847
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813992╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЗtrace_0zшtrace_1zШtrace_2zэtrace_3
D
$Э_self_saveable_object_factories"
_generic_user_object
Е
щ	variables
Щtrainable_variables
чregularization_losses
Ч	keras_api
§__call__
+■&call_and_return_all_conditional_losses
 _random_generator
ђ
state_size
јkernel
Јrecurrent_kernel
	љbias
$Ђ_self_saveable_object_factories"
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
УBт
-__inference_dropout_92_layer_call_fn_55812540inputs"Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
УBт
-__inference_dropout_92_layer_call_fn_55812545inputs"Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЃBђ
H__inference_dropout_92_layer_call_and_return_conditional_losses_55812557inputs"Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЃBђ
H__inference_dropout_92_layer_call_and_return_conditional_losses_55812562inputs"Е
б▓ъ
FullArgSpec!
argsџ
jinputs

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ОBн
-__inference_flatten_92_layer_call_fn_55812567inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЫB№
H__inference_flatten_92_layer_call_and_return_conditional_losses_55812573inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
оBМ
,__inference_dense_593_layer_call_fn_55812582inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ыBЬ
G__inference_dense_593_layer_call_and_return_conditional_losses_55812592inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
оBМ
,__inference_dense_577_layer_call_fn_55812601inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ыBЬ
G__inference_dense_577_layer_call_and_return_conditional_losses_55812632inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
оBМ
,__inference_dense_582_layer_call_fn_55812641inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ыBЬ
G__inference_dense_582_layer_call_and_return_conditional_losses_55812672inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
оBМ
,__inference_dense_587_layer_call_fn_55812681inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ыBЬ
G__inference_dense_587_layer_call_and_return_conditional_losses_55812712inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
оBМ
,__inference_dense_592_layer_call_fn_55812721inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ыBЬ
G__inference_dense_592_layer_call_and_return_conditional_losses_55812752inputs"ў
Љ▓Ї
FullArgSpec
argsџ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Љ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
љBЇ
2__inference_forward_lstm_92_layer_call_fn_55812763inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
љBЇ
2__inference_forward_lstm_92_layer_call_fn_55812774inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
јBІ
2__inference_forward_lstm_92_layer_call_fn_55812785inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
јBІ
2__inference_forward_lstm_92_layer_call_fn_55812796inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ФBе
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55812939inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ФBе
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813082inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЕBд
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813225inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЕBд
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813368inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_dict_wrapper
8
І0
ї1
Ї2"
trackable_list_wrapper
8
І0
ї1
Ї2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓnon_trainable_variables
Ѓlayers
ёmetrics
 Ёlayer_regularization_losses
єlayer_metrics
р	variables
Рtrainable_variables
сregularization_losses
т__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
═
Єtrace_0
ѕtrace_12њ
,__inference_lstm_cell_layer_call_fn_55814009
,__inference_lstm_cell_layer_call_fn_55814026│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЄtrace_0zѕtrace_1
Ѓ
Ѕtrace_0
іtrace_12╚
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814058
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814090│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЅtrace_0zіtrace_1
D
$І_self_saveable_object_factories"
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
Џ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЉBј
3__inference_backward_lstm_92_layer_call_fn_55813379inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЉBј
3__inference_backward_lstm_92_layer_call_fn_55813390inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
3__inference_backward_lstm_92_layer_call_fn_55813401inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЈBї
3__inference_backward_lstm_92_layer_call_fn_55813412inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813557inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
гBЕ
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813702inputs_0"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813847inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
фBД
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813992inputs"╩
├▓┐
FullArgSpec:
args2џ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsб

 
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_dict_wrapper
8
ј0
Ј1
љ2"
trackable_list_wrapper
8
ј0
Ј1
љ2"
trackable_list_wrapper
 "
trackable_list_wrapper
И
їnon_trainable_variables
Їlayers
јmetrics
 Јlayer_regularization_losses
љlayer_metrics
щ	variables
Щtrainable_variables
чregularization_losses
§__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
═
Љtrace_0
њtrace_12њ
,__inference_lstm_cell_layer_call_fn_55814107
,__inference_lstm_cell_layer_call_fn_55814124│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЉtrace_0zњtrace_1
Ѓ
Њtrace_0
ћtrace_12╚
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814156
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814188│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЊtrace_0zћtrace_1
D
$Ћ_self_saveable_object_factories"
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
ЁBѓ
,__inference_lstm_cell_layer_call_fn_55814009inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
,__inference_lstm_cell_layer_call_fn_55814026inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
аBЮ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814058inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
аBЮ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814090inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
ЁBѓ
,__inference_lstm_cell_layer_call_fn_55814107inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
,__inference_lstm_cell_layer_call_fn_55814124inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
аBЮ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814156inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
аBЮ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814188inputsstates_0states_1"│
г▓е
FullArgSpec+
args#џ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsб
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_dict_wrapperд
#__inference__wrapped_model_55808506""#ЃёЁєЄѕЅіІїЇјЈљђЂ+б(
!б
і
input_3
ф ",ф)
'
	dense_593і
	dense_593у
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813557ћјЈљOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "9б6
/і,
tensor_0                  
џ у
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813702ћјЈљOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "9б6
/і,
tensor_0                  
џ ж
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813847ќјЈљQбN
GбD
6і3
inputs'                           

 
p

 
ф "9б6
/і,
tensor_0                  
џ ж
N__inference_backward_lstm_92_layer_call_and_return_conditional_losses_55813992ќјЈљQбN
GбD
6і3
inputs'                           

 
p 

 
ф "9б6
/і,
tensor_0                  
џ ┴
3__inference_backward_lstm_92_layer_call_fn_55813379ЅјЈљOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ".і+
unknown                  ┴
3__inference_backward_lstm_92_layer_call_fn_55813390ЅјЈљOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ".і+
unknown                  ├
3__inference_backward_lstm_92_layer_call_fn_55813401ІјЈљQбN
GбD
6і3
inputs'                           

 
p

 
ф ".і+
unknown                  ├
3__inference_backward_lstm_92_layer_call_fn_55813412ІјЈљQбN
GбD
6і3
inputs'                           

 
p 

 
ф ".і+
unknown                  Щ
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55811671ДІїЇјЈљ\бY
RбO
=џ:
8і5
inputs_0'                           
p

 

 

 
ф "9б6
/і,
tensor_0                  
џ Щ
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55811959ДІїЇјЈљ\бY
RбO
=џ:
8і5
inputs_0'                           
p 

 

 

 
ф "9б6
/і,
tensor_0                  
џ ┼
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55812247sІїЇјЈљ:б7
0б-
і
inputs
p

 

 

 
ф "'б$
і
tensor_0
џ ┼
N__inference_bidirectional_92_layer_call_and_return_conditional_losses_55812535sІїЇјЈљ:б7
0б-
і
inputs
p 

 

 

 
ф "'б$
і
tensor_0
џ н
3__inference_bidirectional_92_layer_call_fn_55811332юІїЇјЈљ\бY
RбO
=џ:
8і5
inputs_0'                           
p

 

 

 
ф ".і+
unknown                  н
3__inference_bidirectional_92_layer_call_fn_55811349юІїЇјЈљ\бY
RбO
=џ:
8і5
inputs_0'                           
p 

 

 

 
ф ".і+
unknown                  Ъ
3__inference_bidirectional_92_layer_call_fn_55811366hІїЇјЈљ:б7
0б-
і
inputs
p

 

 

 
ф "і
unknownЪ
3__inference_bidirectional_92_layer_call_fn_55811383hІїЇјЈљ:б7
0б-
і
inputs
p 

 

 

 
ф "і
unknownц
G__inference_dense_576_layer_call_and_return_conditional_losses_55811243Y"#*б'
 б
і
inputs
ф "'б$
і
tensor_0
џ ~
,__inference_dense_576_layer_call_fn_55811228N"#*б'
 б
і
inputs
ф "і
unknownИ
G__inference_dense_577_layer_call_and_return_conditional_losses_55812632mЃё3б0
)б&
$і!
inputs         
ф "0б-
&і#
tensor_0         
џ њ
,__inference_dense_577_layer_call_fn_55812601bЃё3б0
)б&
$і!
inputs         
ф "%і"
unknown         И
G__inference_dense_582_layer_call_and_return_conditional_losses_55812672mЁє3б0
)б&
$і!
inputs         
ф "0б-
&і#
tensor_0         
џ њ
,__inference_dense_582_layer_call_fn_55812641bЁє3б0
)б&
$і!
inputs         
ф "%і"
unknown         И
G__inference_dense_587_layer_call_and_return_conditional_losses_55812712mЄѕ3б0
)б&
$і!
inputs         
ф "0б-
&і#
tensor_0         
џ њ
,__inference_dense_587_layer_call_fn_55812681bЄѕ3б0
)б&
$і!
inputs         
ф "%і"
unknown         И
G__inference_dense_592_layer_call_and_return_conditional_losses_55812752mЅі3б0
)б&
$і!
inputs         
ф "0б-
&і#
tensor_0         
џ њ
,__inference_dense_592_layer_call_fn_55812721bЅі3б0
)б&
$і!
inputs         
ф "%і"
unknown         Ъ
G__inference_dense_593_layer_call_and_return_conditional_losses_55812592TђЂ'б$
б
і
inputs	ђ
ф "#б 
і
tensor_0
џ y
,__inference_dense_593_layer_call_fn_55812582IђЂ'б$
б
і
inputs	ђ
ф "і
unknownЦ
H__inference_dropout_92_layer_call_and_return_conditional_losses_55812557Y.б+
$б!
і
inputs
p
ф "'б$
і
tensor_0
џ Ц
H__inference_dropout_92_layer_call_and_return_conditional_losses_55812562Y.б+
$б!
і
inputs
p 
ф "'б$
і
tensor_0
џ 
-__inference_dropout_92_layer_call_fn_55812540N.б+
$б!
і
inputs
p
ф "і
unknown
-__inference_dropout_92_layer_call_fn_55812545N.б+
$б!
і
inputs
p 
ф "і
unknownъ
H__inference_flatten_92_layer_call_and_return_conditional_losses_55812573R*б'
 б
і
inputs
ф "$б!
і
tensor_0	ђ
џ x
-__inference_flatten_92_layer_call_fn_55812567G*б'
 б
і
inputs
ф "і
unknown	ђТ
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55812939ћІїЇOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф "9б6
/і,
tensor_0                  
џ Т
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813082ћІїЇOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф "9б6
/і,
tensor_0                  
џ У
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813225ќІїЇQбN
GбD
6і3
inputs'                           

 
p

 
ф "9б6
/і,
tensor_0                  
џ У
M__inference_forward_lstm_92_layer_call_and_return_conditional_losses_55813368ќІїЇQбN
GбD
6і3
inputs'                           

 
p 

 
ф "9б6
/і,
tensor_0                  
џ └
2__inference_forward_lstm_92_layer_call_fn_55812763ЅІїЇOбL
EбB
4џ1
/і,
inputs_0                  

 
p

 
ф ".і+
unknown                  └
2__inference_forward_lstm_92_layer_call_fn_55812774ЅІїЇOбL
EбB
4џ1
/і,
inputs_0                  

 
p 

 
ф ".і+
unknown                  ┬
2__inference_forward_lstm_92_layer_call_fn_55812785ІІїЇQбN
GбD
6і3
inputs'                           

 
p

 
ф ".і+
unknown                  ┬
2__inference_forward_lstm_92_layer_call_fn_55812796ІІїЇQбN
GбD
6і3
inputs'                           

 
p 

 
ф ".і+
unknown                  М
G__inference_lambda_96_layer_call_and_return_conditional_losses_55811261Є\бY
RбO
EџB
і
inputs_0
!і
inputs_1

 
p
ф "'б$
і
tensor_0
џ М
G__inference_lambda_96_layer_call_and_return_conditional_losses_55811267Є\бY
RбO
EџB
і
inputs_0
!і
inputs_1

 
p 
ф "'б$
і
tensor_0
џ г
,__inference_lambda_96_layer_call_fn_55811249|\бY
RбO
EџB
і
inputs_0
!і
inputs_1

 
p
ф "і
unknownг
,__inference_lambda_96_layer_call_fn_55811255|\бY
RбO
EџB
і
inputs_0
!і
inputs_1

 
p 
ф "і
unknownМ
G__inference_lambda_97_layer_call_and_return_conditional_losses_55811285ЄXбU
NбK
Aџ>
і
inputs_0
і
inputs_1

 
p
ф "+б(
!і
tensor_0
џ М
G__inference_lambda_97_layer_call_and_return_conditional_losses_55811291ЄXбU
NбK
Aџ>
і
inputs_0
і
inputs_1

 
p 
ф "+б(
!і
tensor_0
џ г
,__inference_lambda_97_layer_call_fn_55811273|XбU
NбK
Aџ>
і
inputs_0
і
inputs_1

 
p
ф " і
unknownг
,__inference_lambda_97_layer_call_fn_55811279|XбU
NбK
Aџ>
і
inputs_0
і
inputs_1

 
p 
ф " і
unknownМ
G__inference_lambda_98_layer_call_and_return_conditional_losses_55811309Є\бY
RбO
EџB
!і
inputs_0
і
inputs_1

 
p
ф "'б$
і
tensor_0
џ М
G__inference_lambda_98_layer_call_and_return_conditional_losses_55811315Є\бY
RбO
EџB
!і
inputs_0
і
inputs_1

 
p 
ф "'б$
і
tensor_0
џ г
,__inference_lambda_98_layer_call_fn_55811297|\бY
RбO
EџB
!і
inputs_0
і
inputs_1

 
p
ф "і
unknownг
,__inference_lambda_98_layer_call_fn_55811303|\бY
RбO
EџB
!і
inputs_0
і
inputs_1

 
p 
ф "і
unknownс
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814058ЌІїЇђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ с
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814090ЌІїЇђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ с
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814156ЌјЈљђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ с
G__inference_lstm_cell_layer_call_and_return_conditional_losses_55814188ЌјЈљђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "ЅбЁ
~б{
$і!

tensor_0_0         
SџP
&і#
tensor_0_1_0         
&і#
tensor_0_1_1         
џ Х
,__inference_lstm_cell_layer_call_fn_55814009ЁІїЇђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         Х
,__inference_lstm_cell_layer_call_fn_55814026ЁІїЇђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         Х
,__inference_lstm_cell_layer_call_fn_55814107ЁјЈљђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         Х
,__inference_lstm_cell_layer_call_fn_55814124ЁјЈљђб}
vбs
 і
inputs         
KбH
"і
states_0         
"і
states_1         
p 
ф "xбu
"і
tensor_0         
OџL
$і!

tensor_1_0         
$і!

tensor_1_1         ╬
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808546~ЃёDбA
:б7
-і*
dense_577_input         
p

 
ф "0б-
&і#
tensor_0         
џ ╬
L__inference_sequential_392_layer_call_and_return_conditional_losses_55808555~ЃёDбA
:б7
-і*
dense_577_input         
p 

 
ф "0б-
&і#
tensor_0         
џ е
1__inference_sequential_392_layer_call_fn_55808564sЃёDбA
:б7
-і*
dense_577_input         
p

 
ф "%і"
unknown         е
1__inference_sequential_392_layer_call_fn_55808573sЃёDбA
:б7
-і*
dense_577_input         
p 

 
ф "%і"
unknown         ╬
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808622~ЁєDбA
:б7
-і*
dense_582_input         
p

 
ф "0б-
&і#
tensor_0         
џ ╬
L__inference_sequential_397_layer_call_and_return_conditional_losses_55808631~ЁєDбA
:б7
-і*
dense_582_input         
p 

 
ф "0б-
&і#
tensor_0         
џ е
1__inference_sequential_397_layer_call_fn_55808640sЁєDбA
:б7
-і*
dense_582_input         
p

 
ф "%і"
unknown         е
1__inference_sequential_397_layer_call_fn_55808649sЁєDбA
:б7
-і*
dense_582_input         
p 

 
ф "%і"
unknown         ╬
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808698~ЄѕDбA
:б7
-і*
dense_587_input         
p

 
ф "0б-
&і#
tensor_0         
џ ╬
L__inference_sequential_402_layer_call_and_return_conditional_losses_55808707~ЄѕDбA
:б7
-і*
dense_587_input         
p 

 
ф "0б-
&і#
tensor_0         
џ е
1__inference_sequential_402_layer_call_fn_55808716sЄѕDбA
:б7
-і*
dense_587_input         
p

 
ф "%і"
unknown         е
1__inference_sequential_402_layer_call_fn_55808725sЄѕDбA
:б7
-і*
dense_587_input         
p 

 
ф "%і"
unknown         ╬
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808774~ЅіDбA
:б7
-і*
dense_592_input         
p

 
ф "0б-
&і#
tensor_0         
џ ╬
L__inference_sequential_407_layer_call_and_return_conditional_losses_55808783~ЅіDбA
:б7
-і*
dense_592_input         
p 

 
ф "0б-
&і#
tensor_0         
џ е
1__inference_sequential_407_layer_call_fn_55808792sЅіDбA
:б7
-і*
dense_592_input         
p

 
ф "%і"
unknown         е
1__inference_sequential_407_layer_call_fn_55808801sЅіDбA
:б7
-і*
dense_592_input         
p 

 
ф "%і"
unknown         х
&__inference_signature_wrapper_55811219і""#ЃёЁєЄѕЅіІїЇјЈљђЂ6б3
б 
,ф)
'
input_3і
input_3",ф)
'
	dense_593і
	dense_593¤
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810617~""#ЃёЁєЄѕЅіІїЇјЈљђЂ3б0
)б&
і
input_3
p

 
ф "#б 
і
tensor_0
џ ¤
M__inference_topk_bilstm_moe_layer_call_and_return_conditional_losses_55810992~""#ЃёЁєЄѕЅіІїЇјЈљђЂ3б0
)б&
і
input_3
p 

 
ф "#б 
і
tensor_0
џ Е
2__inference_topk_bilstm_moe_layer_call_fn_55811033s""#ЃёЁєЄѕЅіІїЇјЈљђЂ3б0
)б&
і
input_3
p

 
ф "і
unknownЕ
2__inference_topk_bilstm_moe_layer_call_fn_55811074s""#ЃёЁєЄѕЅіІїЇјЈљђЂ3б0
)б&
і
input_3
p 

 
ф "і
unknown