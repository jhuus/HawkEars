??8
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
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
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??1
?
stem_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_namestem_conv/kernel
}
$stem_conv/kernel/Read/ReadVariableOpReadVariableOpstem_conv/kernel*&
_output_shapes
: *
dtype0
r
stem_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namestem_bn/gamma
k
!stem_bn/gamma/Read/ReadVariableOpReadVariableOpstem_bn/gamma*
_output_shapes
: *
dtype0
p
stem_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namestem_bn/beta
i
 stem_bn/beta/Read/ReadVariableOpReadVariableOpstem_bn/beta*
_output_shapes
: *
dtype0
~
stem_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namestem_bn/moving_mean
w
'stem_bn/moving_mean/Read/ReadVariableOpReadVariableOpstem_bn/moving_mean*
_output_shapes
: *
dtype0
?
stem_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_namestem_bn/moving_variance

+stem_bn/moving_variance/Read/ReadVariableOpReadVariableOpstem_bn/moving_variance*
_output_shapes
: *
dtype0
?
&stack_0_block0_MB_dw_/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&stack_0_block0_MB_dw_/depthwise_kernel
?
:stack_0_block0_MB_dw_/depthwise_kernel/Read/ReadVariableOpReadVariableOp&stack_0_block0_MB_dw_/depthwise_kernel*&
_output_shapes
: *
dtype0
?
stack_0_block0_MB_dw_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namestack_0_block0_MB_dw_bn/gamma
?
1stack_0_block0_MB_dw_bn/gamma/Read/ReadVariableOpReadVariableOpstack_0_block0_MB_dw_bn/gamma*
_output_shapes
: *
dtype0
?
stack_0_block0_MB_dw_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namestack_0_block0_MB_dw_bn/beta
?
0stack_0_block0_MB_dw_bn/beta/Read/ReadVariableOpReadVariableOpstack_0_block0_MB_dw_bn/beta*
_output_shapes
: *
dtype0
?
#stack_0_block0_MB_dw_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#stack_0_block0_MB_dw_bn/moving_mean
?
7stack_0_block0_MB_dw_bn/moving_mean/Read/ReadVariableOpReadVariableOp#stack_0_block0_MB_dw_bn/moving_mean*
_output_shapes
: *
dtype0
?
'stack_0_block0_MB_dw_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'stack_0_block0_MB_dw_bn/moving_variance
?
;stack_0_block0_MB_dw_bn/moving_variance/Read/ReadVariableOpReadVariableOp'stack_0_block0_MB_dw_bn/moving_variance*
_output_shapes
: *
dtype0
?
stack_0_block0_se_1_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!stack_0_block0_se_1_conv/kernel
?
3stack_0_block0_se_1_conv/kernel/Read/ReadVariableOpReadVariableOpstack_0_block0_se_1_conv/kernel*&
_output_shapes
: *
dtype0
?
stack_0_block0_se_1_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_0_block0_se_1_conv/bias
?
1stack_0_block0_se_1_conv/bias/Read/ReadVariableOpReadVariableOpstack_0_block0_se_1_conv/bias*
_output_shapes
:*
dtype0
?
stack_0_block0_se_2_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!stack_0_block0_se_2_conv/kernel
?
3stack_0_block0_se_2_conv/kernel/Read/ReadVariableOpReadVariableOpstack_0_block0_se_2_conv/kernel*&
_output_shapes
: *
dtype0
?
stack_0_block0_se_2_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namestack_0_block0_se_2_conv/bias
?
1stack_0_block0_se_2_conv/bias/Read/ReadVariableOpReadVariableOpstack_0_block0_se_2_conv/bias*
_output_shapes
: *
dtype0
?
 stack_0_block0_MB_pw_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" stack_0_block0_MB_pw_conv/kernel
?
4stack_0_block0_MB_pw_conv/kernel/Read/ReadVariableOpReadVariableOp stack_0_block0_MB_pw_conv/kernel*&
_output_shapes
: *
dtype0
?
stack_0_block0_MB_pw_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_0_block0_MB_pw_bn/gamma
?
1stack_0_block0_MB_pw_bn/gamma/Read/ReadVariableOpReadVariableOpstack_0_block0_MB_pw_bn/gamma*
_output_shapes
:*
dtype0
?
stack_0_block0_MB_pw_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namestack_0_block0_MB_pw_bn/beta
?
0stack_0_block0_MB_pw_bn/beta/Read/ReadVariableOpReadVariableOpstack_0_block0_MB_pw_bn/beta*
_output_shapes
:*
dtype0
?
#stack_0_block0_MB_pw_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#stack_0_block0_MB_pw_bn/moving_mean
?
7stack_0_block0_MB_pw_bn/moving_mean/Read/ReadVariableOpReadVariableOp#stack_0_block0_MB_pw_bn/moving_mean*
_output_shapes
:*
dtype0
?
'stack_0_block0_MB_pw_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'stack_0_block0_MB_pw_bn/moving_variance
?
;stack_0_block0_MB_pw_bn/moving_variance/Read/ReadVariableOpReadVariableOp'stack_0_block0_MB_pw_bn/moving_variance*
_output_shapes
:*
dtype0
?
&stack_1_block0_MB_dw_/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&stack_1_block0_MB_dw_/depthwise_kernel
?
:stack_1_block0_MB_dw_/depthwise_kernel/Read/ReadVariableOpReadVariableOp&stack_1_block0_MB_dw_/depthwise_kernel*&
_output_shapes
:*
dtype0
?
stack_1_block0_MB_dw_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_1_block0_MB_dw_bn/gamma
?
1stack_1_block0_MB_dw_bn/gamma/Read/ReadVariableOpReadVariableOpstack_1_block0_MB_dw_bn/gamma*
_output_shapes
:*
dtype0
?
stack_1_block0_MB_dw_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namestack_1_block0_MB_dw_bn/beta
?
0stack_1_block0_MB_dw_bn/beta/Read/ReadVariableOpReadVariableOpstack_1_block0_MB_dw_bn/beta*
_output_shapes
:*
dtype0
?
#stack_1_block0_MB_dw_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#stack_1_block0_MB_dw_bn/moving_mean
?
7stack_1_block0_MB_dw_bn/moving_mean/Read/ReadVariableOpReadVariableOp#stack_1_block0_MB_dw_bn/moving_mean*
_output_shapes
:*
dtype0
?
'stack_1_block0_MB_dw_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'stack_1_block0_MB_dw_bn/moving_variance
?
;stack_1_block0_MB_dw_bn/moving_variance/Read/ReadVariableOpReadVariableOp'stack_1_block0_MB_dw_bn/moving_variance*
_output_shapes
:*
dtype0
?
stack_1_block0_se_1_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!stack_1_block0_se_1_conv/kernel
?
3stack_1_block0_se_1_conv/kernel/Read/ReadVariableOpReadVariableOpstack_1_block0_se_1_conv/kernel*&
_output_shapes
:*
dtype0
?
stack_1_block0_se_1_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_1_block0_se_1_conv/bias
?
1stack_1_block0_se_1_conv/bias/Read/ReadVariableOpReadVariableOpstack_1_block0_se_1_conv/bias*
_output_shapes
:*
dtype0
?
stack_1_block0_se_2_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!stack_1_block0_se_2_conv/kernel
?
3stack_1_block0_se_2_conv/kernel/Read/ReadVariableOpReadVariableOpstack_1_block0_se_2_conv/kernel*&
_output_shapes
:*
dtype0
?
stack_1_block0_se_2_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_1_block0_se_2_conv/bias
?
1stack_1_block0_se_2_conv/bias/Read/ReadVariableOpReadVariableOpstack_1_block0_se_2_conv/bias*
_output_shapes
:*
dtype0
?
 stack_1_block0_MB_pw_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" stack_1_block0_MB_pw_conv/kernel
?
4stack_1_block0_MB_pw_conv/kernel/Read/ReadVariableOpReadVariableOp stack_1_block0_MB_pw_conv/kernel*&
_output_shapes
:*
dtype0
?
stack_1_block0_MB_pw_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_1_block0_MB_pw_bn/gamma
?
1stack_1_block0_MB_pw_bn/gamma/Read/ReadVariableOpReadVariableOpstack_1_block0_MB_pw_bn/gamma*
_output_shapes
:*
dtype0
?
stack_1_block0_MB_pw_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namestack_1_block0_MB_pw_bn/beta
?
0stack_1_block0_MB_pw_bn/beta/Read/ReadVariableOpReadVariableOpstack_1_block0_MB_pw_bn/beta*
_output_shapes
:*
dtype0
?
#stack_1_block0_MB_pw_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#stack_1_block0_MB_pw_bn/moving_mean
?
7stack_1_block0_MB_pw_bn/moving_mean/Read/ReadVariableOpReadVariableOp#stack_1_block0_MB_pw_bn/moving_mean*
_output_shapes
:*
dtype0
?
'stack_1_block0_MB_pw_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'stack_1_block0_MB_pw_bn/moving_variance
?
;stack_1_block0_MB_pw_bn/moving_variance/Read/ReadVariableOpReadVariableOp'stack_1_block0_MB_pw_bn/moving_variance*
_output_shapes
:*
dtype0
?
&stack_1_block1_MB_dw_/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&stack_1_block1_MB_dw_/depthwise_kernel
?
:stack_1_block1_MB_dw_/depthwise_kernel/Read/ReadVariableOpReadVariableOp&stack_1_block1_MB_dw_/depthwise_kernel*&
_output_shapes
:*
dtype0
?
stack_1_block1_MB_dw_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_1_block1_MB_dw_bn/gamma
?
1stack_1_block1_MB_dw_bn/gamma/Read/ReadVariableOpReadVariableOpstack_1_block1_MB_dw_bn/gamma*
_output_shapes
:*
dtype0
?
stack_1_block1_MB_dw_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namestack_1_block1_MB_dw_bn/beta
?
0stack_1_block1_MB_dw_bn/beta/Read/ReadVariableOpReadVariableOpstack_1_block1_MB_dw_bn/beta*
_output_shapes
:*
dtype0
?
#stack_1_block1_MB_dw_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#stack_1_block1_MB_dw_bn/moving_mean
?
7stack_1_block1_MB_dw_bn/moving_mean/Read/ReadVariableOpReadVariableOp#stack_1_block1_MB_dw_bn/moving_mean*
_output_shapes
:*
dtype0
?
'stack_1_block1_MB_dw_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'stack_1_block1_MB_dw_bn/moving_variance
?
;stack_1_block1_MB_dw_bn/moving_variance/Read/ReadVariableOpReadVariableOp'stack_1_block1_MB_dw_bn/moving_variance*
_output_shapes
:*
dtype0
?
stack_1_block1_se_1_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!stack_1_block1_se_1_conv/kernel
?
3stack_1_block1_se_1_conv/kernel/Read/ReadVariableOpReadVariableOpstack_1_block1_se_1_conv/kernel*&
_output_shapes
:*
dtype0
?
stack_1_block1_se_1_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_1_block1_se_1_conv/bias
?
1stack_1_block1_se_1_conv/bias/Read/ReadVariableOpReadVariableOpstack_1_block1_se_1_conv/bias*
_output_shapes
:*
dtype0
?
stack_1_block1_se_2_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!stack_1_block1_se_2_conv/kernel
?
3stack_1_block1_se_2_conv/kernel/Read/ReadVariableOpReadVariableOpstack_1_block1_se_2_conv/kernel*&
_output_shapes
:*
dtype0
?
stack_1_block1_se_2_conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_1_block1_se_2_conv/bias
?
1stack_1_block1_se_2_conv/bias/Read/ReadVariableOpReadVariableOpstack_1_block1_se_2_conv/bias*
_output_shapes
:*
dtype0
?
 stack_1_block1_MB_pw_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" stack_1_block1_MB_pw_conv/kernel
?
4stack_1_block1_MB_pw_conv/kernel/Read/ReadVariableOpReadVariableOp stack_1_block1_MB_pw_conv/kernel*&
_output_shapes
:*
dtype0
?
stack_1_block1_MB_pw_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namestack_1_block1_MB_pw_bn/gamma
?
1stack_1_block1_MB_pw_bn/gamma/Read/ReadVariableOpReadVariableOpstack_1_block1_MB_pw_bn/gamma*
_output_shapes
:*
dtype0
?
stack_1_block1_MB_pw_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namestack_1_block1_MB_pw_bn/beta
?
0stack_1_block1_MB_pw_bn/beta/Read/ReadVariableOpReadVariableOpstack_1_block1_MB_pw_bn/beta*
_output_shapes
:*
dtype0
?
#stack_1_block1_MB_pw_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#stack_1_block1_MB_pw_bn/moving_mean
?
7stack_1_block1_MB_pw_bn/moving_mean/Read/ReadVariableOpReadVariableOp#stack_1_block1_MB_pw_bn/moving_mean*
_output_shapes
:*
dtype0
?
'stack_1_block1_MB_pw_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'stack_1_block1_MB_pw_bn/moving_variance
?
;stack_1_block1_MB_pw_bn/moving_variance/Read/ReadVariableOpReadVariableOp'stack_1_block1_MB_pw_bn/moving_variance*
_output_shapes
:*
dtype0
?
post_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*!
shared_namepost_conv/kernel
~
$post_conv/kernel/Read/ReadVariableOpReadVariableOppost_conv/kernel*'
_output_shapes
:?
*
dtype0
s
post_bn/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*
shared_namepost_bn/gamma
l
!post_bn/gamma/Read/ReadVariableOpReadVariableOppost_bn/gamma*
_output_shapes	
:?
*
dtype0
q
post_bn/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*
shared_namepost_bn/beta
j
 post_bn/beta/Read/ReadVariableOpReadVariableOppost_bn/beta*
_output_shapes	
:?
*
dtype0

post_bn/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*$
shared_namepost_bn/moving_mean
x
'post_bn/moving_mean/Read/ReadVariableOpReadVariableOppost_bn/moving_mean*
_output_shapes	
:?
*
dtype0
?
post_bn/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*(
shared_namepost_bn/moving_variance
?
+post_bn/moving_variance/Read/ReadVariableOpReadVariableOppost_bn/moving_variance*
_output_shapes	
:?
*
dtype0
?
predictions/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*#
shared_namepredictions/kernel
z
&predictions/kernel/Read/ReadVariableOpReadVariableOppredictions/kernel*
_output_shapes
:	?
*
dtype0
x
predictions/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namepredictions/bias
q
$predictions/bias/Read/ReadVariableOpReadVariableOppredictions/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
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
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
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
?
Adam/stem_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/stem_conv/kernel/m
?
+Adam/stem_conv/kernel/m/Read/ReadVariableOpReadVariableOpAdam/stem_conv/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/stem_bn/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/stem_bn/gamma/m
y
(Adam/stem_bn/gamma/m/Read/ReadVariableOpReadVariableOpAdam/stem_bn/gamma/m*
_output_shapes
: *
dtype0
~
Adam/stem_bn/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/stem_bn/beta/m
w
'Adam/stem_bn/beta/m/Read/ReadVariableOpReadVariableOpAdam/stem_bn/beta/m*
_output_shapes
: *
dtype0
?
-Adam/stack_0_block0_MB_dw_/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/stack_0_block0_MB_dw_/depthwise_kernel/m
?
AAdam/stack_0_block0_MB_dw_/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/stack_0_block0_MB_dw_/depthwise_kernel/m*&
_output_shapes
: *
dtype0
?
$Adam/stack_0_block0_MB_dw_bn/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/stack_0_block0_MB_dw_bn/gamma/m
?
8Adam/stack_0_block0_MB_dw_bn/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/stack_0_block0_MB_dw_bn/gamma/m*
_output_shapes
: *
dtype0
?
#Adam/stack_0_block0_MB_dw_bn/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/stack_0_block0_MB_dw_bn/beta/m
?
7Adam/stack_0_block0_MB_dw_bn/beta/m/Read/ReadVariableOpReadVariableOp#Adam/stack_0_block0_MB_dw_bn/beta/m*
_output_shapes
: *
dtype0
?
&Adam/stack_0_block0_se_1_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/stack_0_block0_se_1_conv/kernel/m
?
:Adam/stack_0_block0_se_1_conv/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/stack_0_block0_se_1_conv/kernel/m*&
_output_shapes
: *
dtype0
?
$Adam/stack_0_block0_se_1_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_0_block0_se_1_conv/bias/m
?
8Adam/stack_0_block0_se_1_conv/bias/m/Read/ReadVariableOpReadVariableOp$Adam/stack_0_block0_se_1_conv/bias/m*
_output_shapes
:*
dtype0
?
&Adam/stack_0_block0_se_2_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/stack_0_block0_se_2_conv/kernel/m
?
:Adam/stack_0_block0_se_2_conv/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/stack_0_block0_se_2_conv/kernel/m*&
_output_shapes
: *
dtype0
?
$Adam/stack_0_block0_se_2_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/stack_0_block0_se_2_conv/bias/m
?
8Adam/stack_0_block0_se_2_conv/bias/m/Read/ReadVariableOpReadVariableOp$Adam/stack_0_block0_se_2_conv/bias/m*
_output_shapes
: *
dtype0
?
'Adam/stack_0_block0_MB_pw_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/stack_0_block0_MB_pw_conv/kernel/m
?
;Adam/stack_0_block0_MB_pw_conv/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/stack_0_block0_MB_pw_conv/kernel/m*&
_output_shapes
: *
dtype0
?
$Adam/stack_0_block0_MB_pw_bn/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_0_block0_MB_pw_bn/gamma/m
?
8Adam/stack_0_block0_MB_pw_bn/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/stack_0_block0_MB_pw_bn/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/stack_0_block0_MB_pw_bn/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_0_block0_MB_pw_bn/beta/m
?
7Adam/stack_0_block0_MB_pw_bn/beta/m/Read/ReadVariableOpReadVariableOp#Adam/stack_0_block0_MB_pw_bn/beta/m*
_output_shapes
:*
dtype0
?
-Adam/stack_1_block0_MB_dw_/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/stack_1_block0_MB_dw_/depthwise_kernel/m
?
AAdam/stack_1_block0_MB_dw_/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/stack_1_block0_MB_dw_/depthwise_kernel/m*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block0_MB_dw_bn/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block0_MB_dw_bn/gamma/m
?
8Adam/stack_1_block0_MB_dw_bn/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block0_MB_dw_bn/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/stack_1_block0_MB_dw_bn/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_1_block0_MB_dw_bn/beta/m
?
7Adam/stack_1_block0_MB_dw_bn/beta/m/Read/ReadVariableOpReadVariableOp#Adam/stack_1_block0_MB_dw_bn/beta/m*
_output_shapes
:*
dtype0
?
&Adam/stack_1_block0_se_1_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/stack_1_block0_se_1_conv/kernel/m
?
:Adam/stack_1_block0_se_1_conv/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/stack_1_block0_se_1_conv/kernel/m*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block0_se_1_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block0_se_1_conv/bias/m
?
8Adam/stack_1_block0_se_1_conv/bias/m/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block0_se_1_conv/bias/m*
_output_shapes
:*
dtype0
?
&Adam/stack_1_block0_se_2_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/stack_1_block0_se_2_conv/kernel/m
?
:Adam/stack_1_block0_se_2_conv/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/stack_1_block0_se_2_conv/kernel/m*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block0_se_2_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block0_se_2_conv/bias/m
?
8Adam/stack_1_block0_se_2_conv/bias/m/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block0_se_2_conv/bias/m*
_output_shapes
:*
dtype0
?
'Adam/stack_1_block0_MB_pw_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/stack_1_block0_MB_pw_conv/kernel/m
?
;Adam/stack_1_block0_MB_pw_conv/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/stack_1_block0_MB_pw_conv/kernel/m*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block0_MB_pw_bn/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block0_MB_pw_bn/gamma/m
?
8Adam/stack_1_block0_MB_pw_bn/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block0_MB_pw_bn/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/stack_1_block0_MB_pw_bn/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_1_block0_MB_pw_bn/beta/m
?
7Adam/stack_1_block0_MB_pw_bn/beta/m/Read/ReadVariableOpReadVariableOp#Adam/stack_1_block0_MB_pw_bn/beta/m*
_output_shapes
:*
dtype0
?
-Adam/stack_1_block1_MB_dw_/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/stack_1_block1_MB_dw_/depthwise_kernel/m
?
AAdam/stack_1_block1_MB_dw_/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp-Adam/stack_1_block1_MB_dw_/depthwise_kernel/m*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block1_MB_dw_bn/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block1_MB_dw_bn/gamma/m
?
8Adam/stack_1_block1_MB_dw_bn/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block1_MB_dw_bn/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/stack_1_block1_MB_dw_bn/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_1_block1_MB_dw_bn/beta/m
?
7Adam/stack_1_block1_MB_dw_bn/beta/m/Read/ReadVariableOpReadVariableOp#Adam/stack_1_block1_MB_dw_bn/beta/m*
_output_shapes
:*
dtype0
?
&Adam/stack_1_block1_se_1_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/stack_1_block1_se_1_conv/kernel/m
?
:Adam/stack_1_block1_se_1_conv/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/stack_1_block1_se_1_conv/kernel/m*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block1_se_1_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block1_se_1_conv/bias/m
?
8Adam/stack_1_block1_se_1_conv/bias/m/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block1_se_1_conv/bias/m*
_output_shapes
:*
dtype0
?
&Adam/stack_1_block1_se_2_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/stack_1_block1_se_2_conv/kernel/m
?
:Adam/stack_1_block1_se_2_conv/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/stack_1_block1_se_2_conv/kernel/m*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block1_se_2_conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block1_se_2_conv/bias/m
?
8Adam/stack_1_block1_se_2_conv/bias/m/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block1_se_2_conv/bias/m*
_output_shapes
:*
dtype0
?
'Adam/stack_1_block1_MB_pw_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/stack_1_block1_MB_pw_conv/kernel/m
?
;Adam/stack_1_block1_MB_pw_conv/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/stack_1_block1_MB_pw_conv/kernel/m*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block1_MB_pw_bn/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block1_MB_pw_bn/gamma/m
?
8Adam/stack_1_block1_MB_pw_bn/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block1_MB_pw_bn/gamma/m*
_output_shapes
:*
dtype0
?
#Adam/stack_1_block1_MB_pw_bn/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_1_block1_MB_pw_bn/beta/m
?
7Adam/stack_1_block1_MB_pw_bn/beta/m/Read/ReadVariableOpReadVariableOp#Adam/stack_1_block1_MB_pw_bn/beta/m*
_output_shapes
:*
dtype0
?
Adam/post_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*(
shared_nameAdam/post_conv/kernel/m
?
+Adam/post_conv/kernel/m/Read/ReadVariableOpReadVariableOpAdam/post_conv/kernel/m*'
_output_shapes
:?
*
dtype0
?
Adam/post_bn/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*%
shared_nameAdam/post_bn/gamma/m
z
(Adam/post_bn/gamma/m/Read/ReadVariableOpReadVariableOpAdam/post_bn/gamma/m*
_output_shapes	
:?
*
dtype0

Adam/post_bn/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*$
shared_nameAdam/post_bn/beta/m
x
'Adam/post_bn/beta/m/Read/ReadVariableOpReadVariableOpAdam/post_bn/beta/m*
_output_shapes	
:?
*
dtype0
?
Adam/predictions/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
**
shared_nameAdam/predictions/kernel/m
?
-Adam/predictions/kernel/m/Read/ReadVariableOpReadVariableOpAdam/predictions/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/predictions/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/predictions/bias/m

+Adam/predictions/bias/m/Read/ReadVariableOpReadVariableOpAdam/predictions/bias/m*
_output_shapes
:*
dtype0
?
Adam/stem_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/stem_conv/kernel/v
?
+Adam/stem_conv/kernel/v/Read/ReadVariableOpReadVariableOpAdam/stem_conv/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/stem_bn/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/stem_bn/gamma/v
y
(Adam/stem_bn/gamma/v/Read/ReadVariableOpReadVariableOpAdam/stem_bn/gamma/v*
_output_shapes
: *
dtype0
~
Adam/stem_bn/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/stem_bn/beta/v
w
'Adam/stem_bn/beta/v/Read/ReadVariableOpReadVariableOpAdam/stem_bn/beta/v*
_output_shapes
: *
dtype0
?
-Adam/stack_0_block0_MB_dw_/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/stack_0_block0_MB_dw_/depthwise_kernel/v
?
AAdam/stack_0_block0_MB_dw_/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/stack_0_block0_MB_dw_/depthwise_kernel/v*&
_output_shapes
: *
dtype0
?
$Adam/stack_0_block0_MB_dw_bn/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/stack_0_block0_MB_dw_bn/gamma/v
?
8Adam/stack_0_block0_MB_dw_bn/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/stack_0_block0_MB_dw_bn/gamma/v*
_output_shapes
: *
dtype0
?
#Adam/stack_0_block0_MB_dw_bn/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/stack_0_block0_MB_dw_bn/beta/v
?
7Adam/stack_0_block0_MB_dw_bn/beta/v/Read/ReadVariableOpReadVariableOp#Adam/stack_0_block0_MB_dw_bn/beta/v*
_output_shapes
: *
dtype0
?
&Adam/stack_0_block0_se_1_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/stack_0_block0_se_1_conv/kernel/v
?
:Adam/stack_0_block0_se_1_conv/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/stack_0_block0_se_1_conv/kernel/v*&
_output_shapes
: *
dtype0
?
$Adam/stack_0_block0_se_1_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_0_block0_se_1_conv/bias/v
?
8Adam/stack_0_block0_se_1_conv/bias/v/Read/ReadVariableOpReadVariableOp$Adam/stack_0_block0_se_1_conv/bias/v*
_output_shapes
:*
dtype0
?
&Adam/stack_0_block0_se_2_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/stack_0_block0_se_2_conv/kernel/v
?
:Adam/stack_0_block0_se_2_conv/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/stack_0_block0_se_2_conv/kernel/v*&
_output_shapes
: *
dtype0
?
$Adam/stack_0_block0_se_2_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$Adam/stack_0_block0_se_2_conv/bias/v
?
8Adam/stack_0_block0_se_2_conv/bias/v/Read/ReadVariableOpReadVariableOp$Adam/stack_0_block0_se_2_conv/bias/v*
_output_shapes
: *
dtype0
?
'Adam/stack_0_block0_MB_pw_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'Adam/stack_0_block0_MB_pw_conv/kernel/v
?
;Adam/stack_0_block0_MB_pw_conv/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/stack_0_block0_MB_pw_conv/kernel/v*&
_output_shapes
: *
dtype0
?
$Adam/stack_0_block0_MB_pw_bn/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_0_block0_MB_pw_bn/gamma/v
?
8Adam/stack_0_block0_MB_pw_bn/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/stack_0_block0_MB_pw_bn/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/stack_0_block0_MB_pw_bn/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_0_block0_MB_pw_bn/beta/v
?
7Adam/stack_0_block0_MB_pw_bn/beta/v/Read/ReadVariableOpReadVariableOp#Adam/stack_0_block0_MB_pw_bn/beta/v*
_output_shapes
:*
dtype0
?
-Adam/stack_1_block0_MB_dw_/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/stack_1_block0_MB_dw_/depthwise_kernel/v
?
AAdam/stack_1_block0_MB_dw_/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/stack_1_block0_MB_dw_/depthwise_kernel/v*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block0_MB_dw_bn/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block0_MB_dw_bn/gamma/v
?
8Adam/stack_1_block0_MB_dw_bn/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block0_MB_dw_bn/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/stack_1_block0_MB_dw_bn/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_1_block0_MB_dw_bn/beta/v
?
7Adam/stack_1_block0_MB_dw_bn/beta/v/Read/ReadVariableOpReadVariableOp#Adam/stack_1_block0_MB_dw_bn/beta/v*
_output_shapes
:*
dtype0
?
&Adam/stack_1_block0_se_1_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/stack_1_block0_se_1_conv/kernel/v
?
:Adam/stack_1_block0_se_1_conv/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/stack_1_block0_se_1_conv/kernel/v*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block0_se_1_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block0_se_1_conv/bias/v
?
8Adam/stack_1_block0_se_1_conv/bias/v/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block0_se_1_conv/bias/v*
_output_shapes
:*
dtype0
?
&Adam/stack_1_block0_se_2_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/stack_1_block0_se_2_conv/kernel/v
?
:Adam/stack_1_block0_se_2_conv/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/stack_1_block0_se_2_conv/kernel/v*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block0_se_2_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block0_se_2_conv/bias/v
?
8Adam/stack_1_block0_se_2_conv/bias/v/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block0_se_2_conv/bias/v*
_output_shapes
:*
dtype0
?
'Adam/stack_1_block0_MB_pw_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/stack_1_block0_MB_pw_conv/kernel/v
?
;Adam/stack_1_block0_MB_pw_conv/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/stack_1_block0_MB_pw_conv/kernel/v*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block0_MB_pw_bn/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block0_MB_pw_bn/gamma/v
?
8Adam/stack_1_block0_MB_pw_bn/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block0_MB_pw_bn/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/stack_1_block0_MB_pw_bn/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_1_block0_MB_pw_bn/beta/v
?
7Adam/stack_1_block0_MB_pw_bn/beta/v/Read/ReadVariableOpReadVariableOp#Adam/stack_1_block0_MB_pw_bn/beta/v*
_output_shapes
:*
dtype0
?
-Adam/stack_1_block1_MB_dw_/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*>
shared_name/-Adam/stack_1_block1_MB_dw_/depthwise_kernel/v
?
AAdam/stack_1_block1_MB_dw_/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp-Adam/stack_1_block1_MB_dw_/depthwise_kernel/v*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block1_MB_dw_bn/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block1_MB_dw_bn/gamma/v
?
8Adam/stack_1_block1_MB_dw_bn/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block1_MB_dw_bn/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/stack_1_block1_MB_dw_bn/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_1_block1_MB_dw_bn/beta/v
?
7Adam/stack_1_block1_MB_dw_bn/beta/v/Read/ReadVariableOpReadVariableOp#Adam/stack_1_block1_MB_dw_bn/beta/v*
_output_shapes
:*
dtype0
?
&Adam/stack_1_block1_se_1_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/stack_1_block1_se_1_conv/kernel/v
?
:Adam/stack_1_block1_se_1_conv/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/stack_1_block1_se_1_conv/kernel/v*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block1_se_1_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block1_se_1_conv/bias/v
?
8Adam/stack_1_block1_se_1_conv/bias/v/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block1_se_1_conv/bias/v*
_output_shapes
:*
dtype0
?
&Adam/stack_1_block1_se_2_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/stack_1_block1_se_2_conv/kernel/v
?
:Adam/stack_1_block1_se_2_conv/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/stack_1_block1_se_2_conv/kernel/v*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block1_se_2_conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block1_se_2_conv/bias/v
?
8Adam/stack_1_block1_se_2_conv/bias/v/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block1_se_2_conv/bias/v*
_output_shapes
:*
dtype0
?
'Adam/stack_1_block1_MB_pw_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'Adam/stack_1_block1_MB_pw_conv/kernel/v
?
;Adam/stack_1_block1_MB_pw_conv/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/stack_1_block1_MB_pw_conv/kernel/v*&
_output_shapes
:*
dtype0
?
$Adam/stack_1_block1_MB_pw_bn/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/stack_1_block1_MB_pw_bn/gamma/v
?
8Adam/stack_1_block1_MB_pw_bn/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/stack_1_block1_MB_pw_bn/gamma/v*
_output_shapes
:*
dtype0
?
#Adam/stack_1_block1_MB_pw_bn/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/stack_1_block1_MB_pw_bn/beta/v
?
7Adam/stack_1_block1_MB_pw_bn/beta/v/Read/ReadVariableOpReadVariableOp#Adam/stack_1_block1_MB_pw_bn/beta/v*
_output_shapes
:*
dtype0
?
Adam/post_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*(
shared_nameAdam/post_conv/kernel/v
?
+Adam/post_conv/kernel/v/Read/ReadVariableOpReadVariableOpAdam/post_conv/kernel/v*'
_output_shapes
:?
*
dtype0
?
Adam/post_bn/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*%
shared_nameAdam/post_bn/gamma/v
z
(Adam/post_bn/gamma/v/Read/ReadVariableOpReadVariableOpAdam/post_bn/gamma/v*
_output_shapes	
:?
*
dtype0

Adam/post_bn/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?
*$
shared_nameAdam/post_bn/beta/v
x
'Adam/post_bn/beta/v/Read/ReadVariableOpReadVariableOpAdam/post_bn/beta/v*
_output_shapes	
:?
*
dtype0
?
Adam/predictions/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
**
shared_nameAdam/predictions/kernel/v
?
-Adam/predictions/kernel/v/Read/ReadVariableOpReadVariableOpAdam/predictions/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/predictions/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/predictions/bias/v

+Adam/predictions/bias/v/Read/ReadVariableOpReadVariableOpAdam/predictions/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ݕ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer-27
layer_with_weights-14
layer-28
layer_with_weights-15
layer-29
layer-30
 layer-31
!layer_with_weights-16
!layer-32
"layer-33
#layer_with_weights-17
#layer-34
$layer-35
%layer-36
&layer_with_weights-18
&layer-37
'layer_with_weights-19
'layer-38
(layer-39
)layer-40
*layer_with_weights-20
*layer-41
+layer_with_weights-21
+layer-42
,layer-43
-layer-44
.layer-45
/layer_with_weights-22
/layer-46
0	optimizer
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5
signatures
 
^

6kernel
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
R
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
h
Hdepthwise_kernel
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
R
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api

Z	keras_api
h

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
R
a	variables
btrainable_variables
cregularization_losses
d	keras_api
h

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
R
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
R
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
^

skernel
t	variables
utrainable_variables
vregularization_losses
w	keras_api
?
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance
}	variables
~trainable_variables
regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
m
?depthwise_kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api

?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
c
?kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
m
?depthwise_kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api

?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
c
?kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
h
?noise_shape
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
c
?kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
?beta_1
?beta_2

?decay
?learning_rate
	?iter6m?<m?=m?Hm?Nm?Om?[m?\m?em?fm?sm?ym?zm?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?6v?<v?=v?Hv?Nv?Ov?[v?\v?ev?fv?sv?yv?zv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
?
60
<1
=2
>3
?4
H5
N6
O7
P8
Q9
[10
\11
e12
f13
s14
y15
z16
{17
|18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?
60
<1
=2
H3
N4
O5
[6
\7
e8
f9
s10
y11
z12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
 
\Z
VARIABLE_VALUEstem_conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

60

60
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
 
XV
VARIABLE_VALUEstem_bn/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEstem_bn/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEstem_bn/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEstem_bn/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
>2
?3

<0
=1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
|z
VARIABLE_VALUE&stack_0_block0_MB_dw_/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

H0

H0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
 
hf
VARIABLE_VALUEstack_0_block0_MB_dw_bn/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEstack_0_block0_MB_dw_bn/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#stack_0_block0_MB_dw_bn/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'stack_0_block0_MB_dw_bn/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
P2
Q3

N0
O1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
 
ki
VARIABLE_VALUEstack_0_block0_se_1_conv/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEstack_0_block0_se_1_conv/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

[0
\1

[0
\1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
ki
VARIABLE_VALUEstack_0_block0_se_2_conv/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEstack_0_block0_se_2_conv/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

e0
f1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
k	variables
ltrainable_variables
mregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
lj
VARIABLE_VALUE stack_0_block0_MB_pw_conv/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE

s0

s0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
 
hf
VARIABLE_VALUEstack_0_block0_MB_pw_bn/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEstack_0_block0_MB_pw_bn/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#stack_0_block0_MB_pw_bn/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'stack_0_block0_MB_pw_bn/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

y0
z1
{2
|3

y0
z1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
}	variables
~trainable_variables
regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
|z
VARIABLE_VALUE&stack_1_block0_MB_dw_/depthwise_kernel@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

?0

?0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
hf
VARIABLE_VALUEstack_1_block0_MB_dw_bn/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEstack_1_block0_MB_dw_bn/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#stack_1_block0_MB_dw_bn/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'stack_1_block0_MB_dw_bn/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
lj
VARIABLE_VALUEstack_1_block0_se_1_conv/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEstack_1_block0_se_1_conv/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
lj
VARIABLE_VALUEstack_1_block0_se_2_conv/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEstack_1_block0_se_2_conv/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
mk
VARIABLE_VALUE stack_1_block0_MB_pw_conv/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE

?0

?0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
ig
VARIABLE_VALUEstack_1_block0_MB_pw_bn/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEstack_1_block0_MB_pw_bn/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#stack_1_block0_MB_pw_bn/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'stack_1_block0_MB_pw_bn/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
}{
VARIABLE_VALUE&stack_1_block1_MB_dw_/depthwise_kernelAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

?0

?0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
ig
VARIABLE_VALUEstack_1_block1_MB_dw_bn/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEstack_1_block1_MB_dw_bn/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#stack_1_block1_MB_dw_bn/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'stack_1_block1_MB_dw_bn/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
lj
VARIABLE_VALUEstack_1_block1_se_1_conv/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEstack_1_block1_se_1_conv/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
lj
VARIABLE_VALUEstack_1_block1_se_2_conv/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEstack_1_block1_se_2_conv/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
mk
VARIABLE_VALUE stack_1_block1_MB_pw_conv/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE

?0

?0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
ig
VARIABLE_VALUEstack_1_block1_MB_pw_bn/gamma6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEstack_1_block1_MB_pw_bn/beta5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE#stack_1_block1_MB_pw_bn/moving_mean<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE'stack_1_block1_MB_pw_bn/moving_variance@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
][
VARIABLE_VALUEpost_conv/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE

?0

?0
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
YW
VARIABLE_VALUEpost_bn/gamma6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEpost_bn/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEpost_bn/moving_mean<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEpost_bn/moving_variance@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
_]
VARIABLE_VALUEpredictions/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEpredictions/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
?
>0
?1
P2
Q3
{4
|5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?
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
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46

?0
?1
 
 
 
 
 
 
 

>0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

P0
Q1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

{0
|1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}
VARIABLE_VALUEAdam/stem_conv/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/stem_bn/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/stem_bn/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/stack_0_block0_MB_dw_/depthwise_kernel/m\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_0_block0_MB_dw_bn/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_0_block0_MB_dw_bn/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_0_block0_se_1_conv/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_0_block0_se_1_conv/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_0_block0_se_2_conv/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_0_block0_se_2_conv/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'Adam/stack_0_block0_MB_pw_conv/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_0_block0_MB_pw_bn/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_0_block0_MB_pw_bn/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/stack_1_block0_MB_dw_/depthwise_kernel/m\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block0_MB_dw_bn/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_1_block0_MB_dw_bn/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_1_block0_se_1_conv/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block0_se_1_conv/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_1_block0_se_2_conv/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block0_se_2_conv/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'Adam/stack_1_block0_MB_pw_conv/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block0_MB_pw_bn/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_1_block0_MB_pw_bn/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/stack_1_block1_MB_dw_/depthwise_kernel/m]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block1_MB_dw_bn/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_1_block1_MB_dw_bn/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_1_block1_se_1_conv/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block1_se_1_conv/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_1_block1_se_2_conv/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block1_se_2_conv/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'Adam/stack_1_block1_MB_pw_conv/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block1_MB_pw_bn/gamma/mRlayer_with_weights-19/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_1_block1_MB_pw_bn/beta/mQlayer_with_weights-19/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/post_conv/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/post_bn/gamma/mRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/post_bn/beta/mQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/predictions/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/predictions/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/stem_conv/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/stem_bn/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/stem_bn/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/stack_0_block0_MB_dw_/depthwise_kernel/v\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_0_block0_MB_dw_bn/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_0_block0_MB_dw_bn/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_0_block0_se_1_conv/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_0_block0_se_1_conv/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_0_block0_se_2_conv/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_0_block0_se_2_conv/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'Adam/stack_0_block0_MB_pw_conv/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_0_block0_MB_pw_bn/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_0_block0_MB_pw_bn/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/stack_1_block0_MB_dw_/depthwise_kernel/v\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block0_MB_dw_bn/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_1_block0_MB_dw_bn/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_1_block0_se_1_conv/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block0_se_1_conv/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_1_block0_se_2_conv/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block0_se_2_conv/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'Adam/stack_1_block0_MB_pw_conv/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block0_MB_pw_bn/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_1_block0_MB_pw_bn/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/stack_1_block1_MB_dw_/depthwise_kernel/v]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block1_MB_dw_bn/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_1_block1_MB_dw_bn/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_1_block1_se_1_conv/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block1_se_1_conv/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE&Adam/stack_1_block1_se_2_conv/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block1_se_2_conv/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE'Adam/stack_1_block1_MB_pw_conv/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adam/stack_1_block1_MB_pw_bn/gamma/vRlayer_with_weights-19/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adam/stack_1_block1_MB_pw_bn/beta/vQlayer_with_weights-19/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/post_conv/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/post_bn/gamma/vRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/post_bn/beta/vQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/predictions/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/predictions/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*0
_output_shapes
:????????? ?*
dtype0*%
shape:????????? ?
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1stem_conv/kernelstem_bn/gammastem_bn/betastem_bn/moving_meanstem_bn/moving_variance&stack_0_block0_MB_dw_/depthwise_kernelstack_0_block0_MB_dw_bn/gammastack_0_block0_MB_dw_bn/beta#stack_0_block0_MB_dw_bn/moving_mean'stack_0_block0_MB_dw_bn/moving_variancestack_0_block0_se_1_conv/kernelstack_0_block0_se_1_conv/biasstack_0_block0_se_2_conv/kernelstack_0_block0_se_2_conv/bias stack_0_block0_MB_pw_conv/kernelstack_0_block0_MB_pw_bn/gammastack_0_block0_MB_pw_bn/beta#stack_0_block0_MB_pw_bn/moving_mean'stack_0_block0_MB_pw_bn/moving_variance&stack_1_block0_MB_dw_/depthwise_kernelstack_1_block0_MB_dw_bn/gammastack_1_block0_MB_dw_bn/beta#stack_1_block0_MB_dw_bn/moving_mean'stack_1_block0_MB_dw_bn/moving_variancestack_1_block0_se_1_conv/kernelstack_1_block0_se_1_conv/biasstack_1_block0_se_2_conv/kernelstack_1_block0_se_2_conv/bias stack_1_block0_MB_pw_conv/kernelstack_1_block0_MB_pw_bn/gammastack_1_block0_MB_pw_bn/beta#stack_1_block0_MB_pw_bn/moving_mean'stack_1_block0_MB_pw_bn/moving_variance&stack_1_block1_MB_dw_/depthwise_kernelstack_1_block1_MB_dw_bn/gammastack_1_block1_MB_dw_bn/beta#stack_1_block1_MB_dw_bn/moving_mean'stack_1_block1_MB_dw_bn/moving_variancestack_1_block1_se_1_conv/kernelstack_1_block1_se_1_conv/biasstack_1_block1_se_2_conv/kernelstack_1_block1_se_2_conv/bias stack_1_block1_MB_pw_conv/kernelstack_1_block1_MB_pw_bn/gammastack_1_block1_MB_pw_bn/beta#stack_1_block1_MB_pw_bn/moving_mean'stack_1_block1_MB_pw_bn/moving_variancepost_conv/kernelpost_bn/gammapost_bn/betapost_bn/moving_meanpost_bn/moving_variancepredictions/kernelpredictions/bias*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_45064
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?<
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$stem_conv/kernel/Read/ReadVariableOp!stem_bn/gamma/Read/ReadVariableOp stem_bn/beta/Read/ReadVariableOp'stem_bn/moving_mean/Read/ReadVariableOp+stem_bn/moving_variance/Read/ReadVariableOp:stack_0_block0_MB_dw_/depthwise_kernel/Read/ReadVariableOp1stack_0_block0_MB_dw_bn/gamma/Read/ReadVariableOp0stack_0_block0_MB_dw_bn/beta/Read/ReadVariableOp7stack_0_block0_MB_dw_bn/moving_mean/Read/ReadVariableOp;stack_0_block0_MB_dw_bn/moving_variance/Read/ReadVariableOp3stack_0_block0_se_1_conv/kernel/Read/ReadVariableOp1stack_0_block0_se_1_conv/bias/Read/ReadVariableOp3stack_0_block0_se_2_conv/kernel/Read/ReadVariableOp1stack_0_block0_se_2_conv/bias/Read/ReadVariableOp4stack_0_block0_MB_pw_conv/kernel/Read/ReadVariableOp1stack_0_block0_MB_pw_bn/gamma/Read/ReadVariableOp0stack_0_block0_MB_pw_bn/beta/Read/ReadVariableOp7stack_0_block0_MB_pw_bn/moving_mean/Read/ReadVariableOp;stack_0_block0_MB_pw_bn/moving_variance/Read/ReadVariableOp:stack_1_block0_MB_dw_/depthwise_kernel/Read/ReadVariableOp1stack_1_block0_MB_dw_bn/gamma/Read/ReadVariableOp0stack_1_block0_MB_dw_bn/beta/Read/ReadVariableOp7stack_1_block0_MB_dw_bn/moving_mean/Read/ReadVariableOp;stack_1_block0_MB_dw_bn/moving_variance/Read/ReadVariableOp3stack_1_block0_se_1_conv/kernel/Read/ReadVariableOp1stack_1_block0_se_1_conv/bias/Read/ReadVariableOp3stack_1_block0_se_2_conv/kernel/Read/ReadVariableOp1stack_1_block0_se_2_conv/bias/Read/ReadVariableOp4stack_1_block0_MB_pw_conv/kernel/Read/ReadVariableOp1stack_1_block0_MB_pw_bn/gamma/Read/ReadVariableOp0stack_1_block0_MB_pw_bn/beta/Read/ReadVariableOp7stack_1_block0_MB_pw_bn/moving_mean/Read/ReadVariableOp;stack_1_block0_MB_pw_bn/moving_variance/Read/ReadVariableOp:stack_1_block1_MB_dw_/depthwise_kernel/Read/ReadVariableOp1stack_1_block1_MB_dw_bn/gamma/Read/ReadVariableOp0stack_1_block1_MB_dw_bn/beta/Read/ReadVariableOp7stack_1_block1_MB_dw_bn/moving_mean/Read/ReadVariableOp;stack_1_block1_MB_dw_bn/moving_variance/Read/ReadVariableOp3stack_1_block1_se_1_conv/kernel/Read/ReadVariableOp1stack_1_block1_se_1_conv/bias/Read/ReadVariableOp3stack_1_block1_se_2_conv/kernel/Read/ReadVariableOp1stack_1_block1_se_2_conv/bias/Read/ReadVariableOp4stack_1_block1_MB_pw_conv/kernel/Read/ReadVariableOp1stack_1_block1_MB_pw_bn/gamma/Read/ReadVariableOp0stack_1_block1_MB_pw_bn/beta/Read/ReadVariableOp7stack_1_block1_MB_pw_bn/moving_mean/Read/ReadVariableOp;stack_1_block1_MB_pw_bn/moving_variance/Read/ReadVariableOp$post_conv/kernel/Read/ReadVariableOp!post_bn/gamma/Read/ReadVariableOp post_bn/beta/Read/ReadVariableOp'post_bn/moving_mean/Read/ReadVariableOp+post_bn/moving_variance/Read/ReadVariableOp&predictions/kernel/Read/ReadVariableOp$predictions/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/stem_conv/kernel/m/Read/ReadVariableOp(Adam/stem_bn/gamma/m/Read/ReadVariableOp'Adam/stem_bn/beta/m/Read/ReadVariableOpAAdam/stack_0_block0_MB_dw_/depthwise_kernel/m/Read/ReadVariableOp8Adam/stack_0_block0_MB_dw_bn/gamma/m/Read/ReadVariableOp7Adam/stack_0_block0_MB_dw_bn/beta/m/Read/ReadVariableOp:Adam/stack_0_block0_se_1_conv/kernel/m/Read/ReadVariableOp8Adam/stack_0_block0_se_1_conv/bias/m/Read/ReadVariableOp:Adam/stack_0_block0_se_2_conv/kernel/m/Read/ReadVariableOp8Adam/stack_0_block0_se_2_conv/bias/m/Read/ReadVariableOp;Adam/stack_0_block0_MB_pw_conv/kernel/m/Read/ReadVariableOp8Adam/stack_0_block0_MB_pw_bn/gamma/m/Read/ReadVariableOp7Adam/stack_0_block0_MB_pw_bn/beta/m/Read/ReadVariableOpAAdam/stack_1_block0_MB_dw_/depthwise_kernel/m/Read/ReadVariableOp8Adam/stack_1_block0_MB_dw_bn/gamma/m/Read/ReadVariableOp7Adam/stack_1_block0_MB_dw_bn/beta/m/Read/ReadVariableOp:Adam/stack_1_block0_se_1_conv/kernel/m/Read/ReadVariableOp8Adam/stack_1_block0_se_1_conv/bias/m/Read/ReadVariableOp:Adam/stack_1_block0_se_2_conv/kernel/m/Read/ReadVariableOp8Adam/stack_1_block0_se_2_conv/bias/m/Read/ReadVariableOp;Adam/stack_1_block0_MB_pw_conv/kernel/m/Read/ReadVariableOp8Adam/stack_1_block0_MB_pw_bn/gamma/m/Read/ReadVariableOp7Adam/stack_1_block0_MB_pw_bn/beta/m/Read/ReadVariableOpAAdam/stack_1_block1_MB_dw_/depthwise_kernel/m/Read/ReadVariableOp8Adam/stack_1_block1_MB_dw_bn/gamma/m/Read/ReadVariableOp7Adam/stack_1_block1_MB_dw_bn/beta/m/Read/ReadVariableOp:Adam/stack_1_block1_se_1_conv/kernel/m/Read/ReadVariableOp8Adam/stack_1_block1_se_1_conv/bias/m/Read/ReadVariableOp:Adam/stack_1_block1_se_2_conv/kernel/m/Read/ReadVariableOp8Adam/stack_1_block1_se_2_conv/bias/m/Read/ReadVariableOp;Adam/stack_1_block1_MB_pw_conv/kernel/m/Read/ReadVariableOp8Adam/stack_1_block1_MB_pw_bn/gamma/m/Read/ReadVariableOp7Adam/stack_1_block1_MB_pw_bn/beta/m/Read/ReadVariableOp+Adam/post_conv/kernel/m/Read/ReadVariableOp(Adam/post_bn/gamma/m/Read/ReadVariableOp'Adam/post_bn/beta/m/Read/ReadVariableOp-Adam/predictions/kernel/m/Read/ReadVariableOp+Adam/predictions/bias/m/Read/ReadVariableOp+Adam/stem_conv/kernel/v/Read/ReadVariableOp(Adam/stem_bn/gamma/v/Read/ReadVariableOp'Adam/stem_bn/beta/v/Read/ReadVariableOpAAdam/stack_0_block0_MB_dw_/depthwise_kernel/v/Read/ReadVariableOp8Adam/stack_0_block0_MB_dw_bn/gamma/v/Read/ReadVariableOp7Adam/stack_0_block0_MB_dw_bn/beta/v/Read/ReadVariableOp:Adam/stack_0_block0_se_1_conv/kernel/v/Read/ReadVariableOp8Adam/stack_0_block0_se_1_conv/bias/v/Read/ReadVariableOp:Adam/stack_0_block0_se_2_conv/kernel/v/Read/ReadVariableOp8Adam/stack_0_block0_se_2_conv/bias/v/Read/ReadVariableOp;Adam/stack_0_block0_MB_pw_conv/kernel/v/Read/ReadVariableOp8Adam/stack_0_block0_MB_pw_bn/gamma/v/Read/ReadVariableOp7Adam/stack_0_block0_MB_pw_bn/beta/v/Read/ReadVariableOpAAdam/stack_1_block0_MB_dw_/depthwise_kernel/v/Read/ReadVariableOp8Adam/stack_1_block0_MB_dw_bn/gamma/v/Read/ReadVariableOp7Adam/stack_1_block0_MB_dw_bn/beta/v/Read/ReadVariableOp:Adam/stack_1_block0_se_1_conv/kernel/v/Read/ReadVariableOp8Adam/stack_1_block0_se_1_conv/bias/v/Read/ReadVariableOp:Adam/stack_1_block0_se_2_conv/kernel/v/Read/ReadVariableOp8Adam/stack_1_block0_se_2_conv/bias/v/Read/ReadVariableOp;Adam/stack_1_block0_MB_pw_conv/kernel/v/Read/ReadVariableOp8Adam/stack_1_block0_MB_pw_bn/gamma/v/Read/ReadVariableOp7Adam/stack_1_block0_MB_pw_bn/beta/v/Read/ReadVariableOpAAdam/stack_1_block1_MB_dw_/depthwise_kernel/v/Read/ReadVariableOp8Adam/stack_1_block1_MB_dw_bn/gamma/v/Read/ReadVariableOp7Adam/stack_1_block1_MB_dw_bn/beta/v/Read/ReadVariableOp:Adam/stack_1_block1_se_1_conv/kernel/v/Read/ReadVariableOp8Adam/stack_1_block1_se_1_conv/bias/v/Read/ReadVariableOp:Adam/stack_1_block1_se_2_conv/kernel/v/Read/ReadVariableOp8Adam/stack_1_block1_se_2_conv/bias/v/Read/ReadVariableOp;Adam/stack_1_block1_MB_pw_conv/kernel/v/Read/ReadVariableOp8Adam/stack_1_block1_MB_pw_bn/gamma/v/Read/ReadVariableOp7Adam/stack_1_block1_MB_pw_bn/beta/v/Read/ReadVariableOp+Adam/post_conv/kernel/v/Read/ReadVariableOp(Adam/post_bn/gamma/v/Read/ReadVariableOp'Adam/post_bn/beta/v/Read/ReadVariableOp-Adam/predictions/kernel/v/Read/ReadVariableOp+Adam/predictions/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_48404
?&
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestem_conv/kernelstem_bn/gammastem_bn/betastem_bn/moving_meanstem_bn/moving_variance&stack_0_block0_MB_dw_/depthwise_kernelstack_0_block0_MB_dw_bn/gammastack_0_block0_MB_dw_bn/beta#stack_0_block0_MB_dw_bn/moving_mean'stack_0_block0_MB_dw_bn/moving_variancestack_0_block0_se_1_conv/kernelstack_0_block0_se_1_conv/biasstack_0_block0_se_2_conv/kernelstack_0_block0_se_2_conv/bias stack_0_block0_MB_pw_conv/kernelstack_0_block0_MB_pw_bn/gammastack_0_block0_MB_pw_bn/beta#stack_0_block0_MB_pw_bn/moving_mean'stack_0_block0_MB_pw_bn/moving_variance&stack_1_block0_MB_dw_/depthwise_kernelstack_1_block0_MB_dw_bn/gammastack_1_block0_MB_dw_bn/beta#stack_1_block0_MB_dw_bn/moving_mean'stack_1_block0_MB_dw_bn/moving_variancestack_1_block0_se_1_conv/kernelstack_1_block0_se_1_conv/biasstack_1_block0_se_2_conv/kernelstack_1_block0_se_2_conv/bias stack_1_block0_MB_pw_conv/kernelstack_1_block0_MB_pw_bn/gammastack_1_block0_MB_pw_bn/beta#stack_1_block0_MB_pw_bn/moving_mean'stack_1_block0_MB_pw_bn/moving_variance&stack_1_block1_MB_dw_/depthwise_kernelstack_1_block1_MB_dw_bn/gammastack_1_block1_MB_dw_bn/beta#stack_1_block1_MB_dw_bn/moving_mean'stack_1_block1_MB_dw_bn/moving_variancestack_1_block1_se_1_conv/kernelstack_1_block1_se_1_conv/biasstack_1_block1_se_2_conv/kernelstack_1_block1_se_2_conv/bias stack_1_block1_MB_pw_conv/kernelstack_1_block1_MB_pw_bn/gammastack_1_block1_MB_pw_bn/beta#stack_1_block1_MB_pw_bn/moving_mean'stack_1_block1_MB_pw_bn/moving_variancepost_conv/kernelpost_bn/gammapost_bn/betapost_bn/moving_meanpost_bn/moving_variancepredictions/kernelpredictions/biasbeta_1beta_2decaylearning_rate	Adam/itertotalcounttotal_1count_1Adam/stem_conv/kernel/mAdam/stem_bn/gamma/mAdam/stem_bn/beta/m-Adam/stack_0_block0_MB_dw_/depthwise_kernel/m$Adam/stack_0_block0_MB_dw_bn/gamma/m#Adam/stack_0_block0_MB_dw_bn/beta/m&Adam/stack_0_block0_se_1_conv/kernel/m$Adam/stack_0_block0_se_1_conv/bias/m&Adam/stack_0_block0_se_2_conv/kernel/m$Adam/stack_0_block0_se_2_conv/bias/m'Adam/stack_0_block0_MB_pw_conv/kernel/m$Adam/stack_0_block0_MB_pw_bn/gamma/m#Adam/stack_0_block0_MB_pw_bn/beta/m-Adam/stack_1_block0_MB_dw_/depthwise_kernel/m$Adam/stack_1_block0_MB_dw_bn/gamma/m#Adam/stack_1_block0_MB_dw_bn/beta/m&Adam/stack_1_block0_se_1_conv/kernel/m$Adam/stack_1_block0_se_1_conv/bias/m&Adam/stack_1_block0_se_2_conv/kernel/m$Adam/stack_1_block0_se_2_conv/bias/m'Adam/stack_1_block0_MB_pw_conv/kernel/m$Adam/stack_1_block0_MB_pw_bn/gamma/m#Adam/stack_1_block0_MB_pw_bn/beta/m-Adam/stack_1_block1_MB_dw_/depthwise_kernel/m$Adam/stack_1_block1_MB_dw_bn/gamma/m#Adam/stack_1_block1_MB_dw_bn/beta/m&Adam/stack_1_block1_se_1_conv/kernel/m$Adam/stack_1_block1_se_1_conv/bias/m&Adam/stack_1_block1_se_2_conv/kernel/m$Adam/stack_1_block1_se_2_conv/bias/m'Adam/stack_1_block1_MB_pw_conv/kernel/m$Adam/stack_1_block1_MB_pw_bn/gamma/m#Adam/stack_1_block1_MB_pw_bn/beta/mAdam/post_conv/kernel/mAdam/post_bn/gamma/mAdam/post_bn/beta/mAdam/predictions/kernel/mAdam/predictions/bias/mAdam/stem_conv/kernel/vAdam/stem_bn/gamma/vAdam/stem_bn/beta/v-Adam/stack_0_block0_MB_dw_/depthwise_kernel/v$Adam/stack_0_block0_MB_dw_bn/gamma/v#Adam/stack_0_block0_MB_dw_bn/beta/v&Adam/stack_0_block0_se_1_conv/kernel/v$Adam/stack_0_block0_se_1_conv/bias/v&Adam/stack_0_block0_se_2_conv/kernel/v$Adam/stack_0_block0_se_2_conv/bias/v'Adam/stack_0_block0_MB_pw_conv/kernel/v$Adam/stack_0_block0_MB_pw_bn/gamma/v#Adam/stack_0_block0_MB_pw_bn/beta/v-Adam/stack_1_block0_MB_dw_/depthwise_kernel/v$Adam/stack_1_block0_MB_dw_bn/gamma/v#Adam/stack_1_block0_MB_dw_bn/beta/v&Adam/stack_1_block0_se_1_conv/kernel/v$Adam/stack_1_block0_se_1_conv/bias/v&Adam/stack_1_block0_se_2_conv/kernel/v$Adam/stack_1_block0_se_2_conv/bias/v'Adam/stack_1_block0_MB_pw_conv/kernel/v$Adam/stack_1_block0_MB_pw_bn/gamma/v#Adam/stack_1_block0_MB_pw_bn/beta/v-Adam/stack_1_block1_MB_dw_/depthwise_kernel/v$Adam/stack_1_block1_MB_dw_bn/gamma/v#Adam/stack_1_block1_MB_dw_bn/beta/v&Adam/stack_1_block1_se_1_conv/kernel/v$Adam/stack_1_block1_se_1_conv/bias/v&Adam/stack_1_block1_se_2_conv/kernel/v$Adam/stack_1_block1_se_2_conv/bias/v'Adam/stack_1_block1_MB_pw_conv/kernel/v$Adam/stack_1_block1_MB_pw_bn/gamma/v#Adam/stack_1_block1_MB_pw_bn/beta/vAdam/post_conv/kernel/vAdam/post_bn/gamma/vAdam/post_bn/beta/vAdam/predictions/kernel/vAdam/predictions/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_48831??+
?
?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_42588

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
S__inference_stack_0_block0_se_1_conv_layer_call_and_return_conditional_losses_42861

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
S
7__inference_stack_1_block1_dropdrop_layer_call_fn_47100

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_43267h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
??
?7
G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820

inputsB
(stem_conv_conv2d_readvariableop_resource: -
stem_bn_readvariableop_resource: /
!stem_bn_readvariableop_1_resource: >
0stem_bn_fusedbatchnormv3_readvariableop_resource: @
2stem_bn_fusedbatchnormv3_readvariableop_1_resource: Q
7stack_0_block0_mb_dw__depthwise_readvariableop_resource: =
/stack_0_block0_mb_dw_bn_readvariableop_resource: ?
1stack_0_block0_mb_dw_bn_readvariableop_1_resource: N
@stack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource: P
Bstack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource: Q
7stack_0_block0_se_1_conv_conv2d_readvariableop_resource: F
8stack_0_block0_se_1_conv_biasadd_readvariableop_resource:Q
7stack_0_block0_se_2_conv_conv2d_readvariableop_resource: F
8stack_0_block0_se_2_conv_biasadd_readvariableop_resource: R
8stack_0_block0_mb_pw_conv_conv2d_readvariableop_resource: =
/stack_0_block0_mb_pw_bn_readvariableop_resource:?
1stack_0_block0_mb_pw_bn_readvariableop_1_resource:N
@stack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource:Q
7stack_1_block0_mb_dw__depthwise_readvariableop_resource:=
/stack_1_block0_mb_dw_bn_readvariableop_resource:?
1stack_1_block0_mb_dw_bn_readvariableop_1_resource:N
@stack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource:Q
7stack_1_block0_se_1_conv_conv2d_readvariableop_resource:F
8stack_1_block0_se_1_conv_biasadd_readvariableop_resource:Q
7stack_1_block0_se_2_conv_conv2d_readvariableop_resource:F
8stack_1_block0_se_2_conv_biasadd_readvariableop_resource:R
8stack_1_block0_mb_pw_conv_conv2d_readvariableop_resource:=
/stack_1_block0_mb_pw_bn_readvariableop_resource:?
1stack_1_block0_mb_pw_bn_readvariableop_1_resource:N
@stack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource:Q
7stack_1_block1_mb_dw__depthwise_readvariableop_resource:=
/stack_1_block1_mb_dw_bn_readvariableop_resource:?
1stack_1_block1_mb_dw_bn_readvariableop_1_resource:N
@stack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource:Q
7stack_1_block1_se_1_conv_conv2d_readvariableop_resource:F
8stack_1_block1_se_1_conv_biasadd_readvariableop_resource:Q
7stack_1_block1_se_2_conv_conv2d_readvariableop_resource:F
8stack_1_block1_se_2_conv_biasadd_readvariableop_resource:R
8stack_1_block1_mb_pw_conv_conv2d_readvariableop_resource:=
/stack_1_block1_mb_pw_bn_readvariableop_resource:?
1stack_1_block1_mb_pw_bn_readvariableop_1_resource:N
@stack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource:C
(post_conv_conv2d_readvariableop_resource:?
.
post_bn_readvariableop_resource:	?
0
!post_bn_readvariableop_1_resource:	?
?
0post_bn_fusedbatchnormv3_readvariableop_resource:	?
A
2post_bn_fusedbatchnormv3_readvariableop_1_resource:	?
=
*predictions_matmul_readvariableop_resource:	?
9
+predictions_biasadd_readvariableop_resource:
identity??post_bn/AssignNewValue?post_bn/AssignNewValue_1?'post_bn/FusedBatchNormV3/ReadVariableOp?)post_bn/FusedBatchNormV3/ReadVariableOp_1?post_bn/ReadVariableOp?post_bn/ReadVariableOp_1?post_conv/Conv2D/ReadVariableOp?"predictions/BiasAdd/ReadVariableOp?!predictions/MatMul/ReadVariableOp?.stack_0_block0_MB_dw_/depthwise/ReadVariableOp?&stack_0_block0_MB_dw_bn/AssignNewValue?(stack_0_block0_MB_dw_bn/AssignNewValue_1?7stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp?9stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_0_block0_MB_dw_bn/ReadVariableOp?(stack_0_block0_MB_dw_bn/ReadVariableOp_1?&stack_0_block0_MB_pw_bn/AssignNewValue?(stack_0_block0_MB_pw_bn/AssignNewValue_1?7stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp?9stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_0_block0_MB_pw_bn/ReadVariableOp?(stack_0_block0_MB_pw_bn/ReadVariableOp_1?/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp?/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp?.stack_0_block0_se_1_conv/Conv2D/ReadVariableOp?/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp?.stack_0_block0_se_2_conv/Conv2D/ReadVariableOp?.stack_1_block0_MB_dw_/depthwise/ReadVariableOp?&stack_1_block0_MB_dw_bn/AssignNewValue?(stack_1_block0_MB_dw_bn/AssignNewValue_1?7stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp?9stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_1_block0_MB_dw_bn/ReadVariableOp?(stack_1_block0_MB_dw_bn/ReadVariableOp_1?&stack_1_block0_MB_pw_bn/AssignNewValue?(stack_1_block0_MB_pw_bn/AssignNewValue_1?7stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp?9stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_1_block0_MB_pw_bn/ReadVariableOp?(stack_1_block0_MB_pw_bn/ReadVariableOp_1?/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp?/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp?.stack_1_block0_se_1_conv/Conv2D/ReadVariableOp?/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp?.stack_1_block0_se_2_conv/Conv2D/ReadVariableOp?.stack_1_block1_MB_dw_/depthwise/ReadVariableOp?&stack_1_block1_MB_dw_bn/AssignNewValue?(stack_1_block1_MB_dw_bn/AssignNewValue_1?7stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp?9stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_1_block1_MB_dw_bn/ReadVariableOp?(stack_1_block1_MB_dw_bn/ReadVariableOp_1?&stack_1_block1_MB_pw_bn/AssignNewValue?(stack_1_block1_MB_pw_bn/AssignNewValue_1?7stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp?9stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_1_block1_MB_pw_bn/ReadVariableOp?(stack_1_block1_MB_pw_bn/ReadVariableOp_1?/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp?/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp?.stack_1_block1_se_1_conv/Conv2D/ReadVariableOp?/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp?.stack_1_block1_se_2_conv/Conv2D/ReadVariableOp?stem_bn/AssignNewValue?stem_bn/AssignNewValue_1?'stem_bn/FusedBatchNormV3/ReadVariableOp?)stem_bn/FusedBatchNormV3/ReadVariableOp_1?stem_bn/ReadVariableOp?stem_bn/ReadVariableOp_1?stem_conv/Conv2D/ReadVariableOp?
stem_conv/Conv2D/ReadVariableOpReadVariableOp(stem_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
stem_conv/Conv2DConv2Dinputs'stem_conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
stem_bn/ReadVariableOpReadVariableOpstem_bn_readvariableop_resource*
_output_shapes
: *
dtype0v
stem_bn/ReadVariableOp_1ReadVariableOp!stem_bn_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'stem_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp0stem_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
)stem_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp2stem_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
stem_bn/FusedBatchNormV3FusedBatchNormV3stem_conv/Conv2D:output:0stem_bn/ReadVariableOp:value:0 stem_bn/ReadVariableOp_1:value:0/stem_bn/FusedBatchNormV3/ReadVariableOp:value:01stem_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
stem_bn/AssignNewValueAssignVariableOp0stem_bn_fusedbatchnormv3_readvariableop_resource%stem_bn/FusedBatchNormV3:batch_mean:0(^stem_bn/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
stem_bn/AssignNewValue_1AssignVariableOp2stem_bn_fusedbatchnormv3_readvariableop_1_resource)stem_bn/FusedBatchNormV3:batch_variance:0*^stem_bn/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0v
stem_swish/SigmoidSigmoidstem_bn/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????? ?
stem_swish/mulMulstem_bn/FusedBatchNormV3:y:0stem_swish/Sigmoid:y:0*
T0*0
_output_shapes
:?????????? n
stem_swish/IdentityIdentitystem_swish/mul:z:0*
T0*0
_output_shapes
:?????????? ?
stem_swish/IdentityN	IdentityNstem_swish/mul:z:0stem_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45564*L
_output_shapes:
8:?????????? :?????????? ?
.stack_0_block0_MB_dw_/depthwise/ReadVariableOpReadVariableOp7stack_0_block0_mb_dw__depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0~
%stack_0_block0_MB_dw_/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             ~
-stack_0_block0_MB_dw_/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
stack_0_block0_MB_dw_/depthwiseDepthwiseConv2dNativestem_swish/IdentityN:output:06stack_0_block0_MB_dw_/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
&stack_0_block0_MB_dw_bn/ReadVariableOpReadVariableOp/stack_0_block0_mb_dw_bn_readvariableop_resource*
_output_shapes
: *
dtype0?
(stack_0_block0_MB_dw_bn/ReadVariableOp_1ReadVariableOp1stack_0_block0_mb_dw_bn_readvariableop_1_resource*
_output_shapes
: *
dtype0?
7stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
9stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
(stack_0_block0_MB_dw_bn/FusedBatchNormV3FusedBatchNormV3(stack_0_block0_MB_dw_/depthwise:output:0.stack_0_block0_MB_dw_bn/ReadVariableOp:value:00stack_0_block0_MB_dw_bn/ReadVariableOp_1:value:0?stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
&stack_0_block0_MB_dw_bn/AssignNewValueAssignVariableOp@stack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource5stack_0_block0_MB_dw_bn/FusedBatchNormV3:batch_mean:08^stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
(stack_0_block0_MB_dw_bn/AssignNewValue_1AssignVariableOpBstack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource9stack_0_block0_MB_dw_bn/FusedBatchNormV3:batch_variance:0:^stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
"stack_0_block0_MB_dw_swish/SigmoidSigmoid,stack_0_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????? ?
stack_0_block0_MB_dw_swish/mulMul,stack_0_block0_MB_dw_bn/FusedBatchNormV3:y:0&stack_0_block0_MB_dw_swish/Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
#stack_0_block0_MB_dw_swish/IdentityIdentity"stack_0_block0_MB_dw_swish/mul:z:0*
T0*0
_output_shapes
:?????????? ?
$stack_0_block0_MB_dw_swish/IdentityN	IdentityN"stack_0_block0_MB_dw_swish/mul:z:0,stack_0_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45589*L
_output_shapes:
8:?????????? :?????????? {
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean/MeanMean-stack_0_block0_MB_dw_swish/IdentityN:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(?
.stack_0_block0_se_1_conv/Conv2D/ReadVariableOpReadVariableOp7stack_0_block0_se_1_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
stack_0_block0_se_1_conv/Conv2DConv2D!tf.math.reduce_mean/Mean:output:06stack_0_block0_se_1_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_0_block0_se_1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_0_block0_se_1_conv/BiasAddBiasAdd(stack_0_block0_se_1_conv/Conv2D:output:07stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
stack_0_block0_se_swish/SigmoidSigmoid)stack_0_block0_se_1_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_0_block0_se_swish/mulMul)stack_0_block0_se_1_conv/BiasAdd:output:0#stack_0_block0_se_swish/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
 stack_0_block0_se_swish/IdentityIdentitystack_0_block0_se_swish/mul:z:0*
T0*/
_output_shapes
:??????????
!stack_0_block0_se_swish/IdentityN	IdentityNstack_0_block0_se_swish/mul:z:0)stack_0_block0_se_1_conv/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-45603*J
_output_shapes8
6:?????????:??????????
.stack_0_block0_se_2_conv/Conv2D/ReadVariableOpReadVariableOp7stack_0_block0_se_2_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
stack_0_block0_se_2_conv/Conv2DConv2D*stack_0_block0_se_swish/IdentityN:output:06stack_0_block0_se_2_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_0_block0_se_2_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
 stack_0_block0_se_2_conv/BiasAddBiasAdd(stack_0_block0_se_2_conv/Conv2D:output:07stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
!stack_0_block0_se_sigmoid/SigmoidSigmoid)stack_0_block0_se_2_conv/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
stack_0_block0_se_out/mulMul-stack_0_block0_MB_dw_swish/IdentityN:output:0%stack_0_block0_se_sigmoid/Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOpReadVariableOp8stack_0_block0_mb_pw_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
 stack_0_block0_MB_pw_conv/Conv2DConv2Dstack_0_block0_se_out/mul:z:07stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
&stack_0_block0_MB_pw_bn/ReadVariableOpReadVariableOp/stack_0_block0_mb_pw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_0_block0_MB_pw_bn/ReadVariableOp_1ReadVariableOp1stack_0_block0_mb_pw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_0_block0_MB_pw_bn/FusedBatchNormV3FusedBatchNormV3)stack_0_block0_MB_pw_conv/Conv2D:output:0.stack_0_block0_MB_pw_bn/ReadVariableOp:value:00stack_0_block0_MB_pw_bn/ReadVariableOp_1:value:0?stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
&stack_0_block0_MB_pw_bn/AssignNewValueAssignVariableOp@stack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource5stack_0_block0_MB_pw_bn/FusedBatchNormV3:batch_mean:08^stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
(stack_0_block0_MB_pw_bn/AssignNewValue_1AssignVariableOpBstack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource9stack_0_block0_MB_pw_bn/FusedBatchNormV3:batch_variance:0:^stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
.stack_1_block0_MB_dw_/depthwise/ReadVariableOpReadVariableOp7stack_1_block0_mb_dw__depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0~
%stack_1_block0_MB_dw_/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            ~
-stack_1_block0_MB_dw_/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
stack_1_block0_MB_dw_/depthwiseDepthwiseConv2dNative,stack_0_block0_MB_pw_bn/FusedBatchNormV3:y:06stack_1_block0_MB_dw_/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
?
&stack_1_block0_MB_dw_bn/ReadVariableOpReadVariableOp/stack_1_block0_mb_dw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_1_block0_MB_dw_bn/ReadVariableOp_1ReadVariableOp1stack_1_block0_mb_dw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_1_block0_MB_dw_bn/FusedBatchNormV3FusedBatchNormV3(stack_1_block0_MB_dw_/depthwise:output:0.stack_1_block0_MB_dw_bn/ReadVariableOp:value:00stack_1_block0_MB_dw_bn/ReadVariableOp_1:value:0?stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
&stack_1_block0_MB_dw_bn/AssignNewValueAssignVariableOp@stack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource5stack_1_block0_MB_dw_bn/FusedBatchNormV3:batch_mean:08^stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
(stack_1_block0_MB_dw_bn/AssignNewValue_1AssignVariableOpBstack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource9stack_1_block0_MB_dw_bn/FusedBatchNormV3:batch_variance:0:^stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
"stack_1_block0_MB_dw_swish/SigmoidSigmoid,stack_1_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????`?
stack_1_block0_MB_dw_swish/mulMul,stack_1_block0_MB_dw_bn/FusedBatchNormV3:y:0&stack_1_block0_MB_dw_swish/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
#stack_1_block0_MB_dw_swish/IdentityIdentity"stack_1_block0_MB_dw_swish/mul:z:0*
T0*/
_output_shapes
:?????????`?
$stack_1_block0_MB_dw_swish/IdentityN	IdentityN"stack_1_block0_MB_dw_swish/mul:z:0,stack_1_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45653*J
_output_shapes8
6:?????????`:?????????`}
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_1/MeanMean-stack_1_block0_MB_dw_swish/IdentityN:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
.stack_1_block0_se_1_conv/Conv2D/ReadVariableOpReadVariableOp7stack_1_block0_se_1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
stack_1_block0_se_1_conv/Conv2DConv2D#tf.math.reduce_mean_1/Mean:output:06stack_1_block0_se_1_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_1_block0_se_1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_1_block0_se_1_conv/BiasAddBiasAdd(stack_1_block0_se_1_conv/Conv2D:output:07stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
stack_1_block0_se_swish/SigmoidSigmoid)stack_1_block0_se_1_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_1_block0_se_swish/mulMul)stack_1_block0_se_1_conv/BiasAdd:output:0#stack_1_block0_se_swish/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
 stack_1_block0_se_swish/IdentityIdentitystack_1_block0_se_swish/mul:z:0*
T0*/
_output_shapes
:??????????
!stack_1_block0_se_swish/IdentityN	IdentityNstack_1_block0_se_swish/mul:z:0)stack_1_block0_se_1_conv/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-45667*J
_output_shapes8
6:?????????:??????????
.stack_1_block0_se_2_conv/Conv2D/ReadVariableOpReadVariableOp7stack_1_block0_se_2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
stack_1_block0_se_2_conv/Conv2DConv2D*stack_1_block0_se_swish/IdentityN:output:06stack_1_block0_se_2_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_1_block0_se_2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_1_block0_se_2_conv/BiasAddBiasAdd(stack_1_block0_se_2_conv/Conv2D:output:07stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
!stack_1_block0_se_sigmoid/SigmoidSigmoid)stack_1_block0_se_2_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_1_block0_se_out/mulMul-stack_1_block0_MB_dw_swish/IdentityN:output:0%stack_1_block0_se_sigmoid/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOpReadVariableOp8stack_1_block0_mb_pw_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
 stack_1_block0_MB_pw_conv/Conv2DConv2Dstack_1_block0_se_out/mul:z:07stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
?
&stack_1_block0_MB_pw_bn/ReadVariableOpReadVariableOp/stack_1_block0_mb_pw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_1_block0_MB_pw_bn/ReadVariableOp_1ReadVariableOp1stack_1_block0_mb_pw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_1_block0_MB_pw_bn/FusedBatchNormV3FusedBatchNormV3)stack_1_block0_MB_pw_conv/Conv2D:output:0.stack_1_block0_MB_pw_bn/ReadVariableOp:value:00stack_1_block0_MB_pw_bn/ReadVariableOp_1:value:0?stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
&stack_1_block0_MB_pw_bn/AssignNewValueAssignVariableOp@stack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource5stack_1_block0_MB_pw_bn/FusedBatchNormV3:batch_mean:08^stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
(stack_1_block0_MB_pw_bn/AssignNewValue_1AssignVariableOpBstack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource9stack_1_block0_MB_pw_bn/FusedBatchNormV3:batch_variance:0:^stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
.stack_1_block1_MB_dw_/depthwise/ReadVariableOpReadVariableOp7stack_1_block1_mb_dw__depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0~
%stack_1_block1_MB_dw_/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            ~
-stack_1_block1_MB_dw_/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
stack_1_block1_MB_dw_/depthwiseDepthwiseConv2dNative,stack_1_block0_MB_pw_bn/FusedBatchNormV3:y:06stack_1_block1_MB_dw_/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
?
&stack_1_block1_MB_dw_bn/ReadVariableOpReadVariableOp/stack_1_block1_mb_dw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_1_block1_MB_dw_bn/ReadVariableOp_1ReadVariableOp1stack_1_block1_mb_dw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_1_block1_MB_dw_bn/FusedBatchNormV3FusedBatchNormV3(stack_1_block1_MB_dw_/depthwise:output:0.stack_1_block1_MB_dw_bn/ReadVariableOp:value:00stack_1_block1_MB_dw_bn/ReadVariableOp_1:value:0?stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
&stack_1_block1_MB_dw_bn/AssignNewValueAssignVariableOp@stack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_resource5stack_1_block1_MB_dw_bn/FusedBatchNormV3:batch_mean:08^stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
(stack_1_block1_MB_dw_bn/AssignNewValue_1AssignVariableOpBstack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource9stack_1_block1_MB_dw_bn/FusedBatchNormV3:batch_variance:0:^stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0?
"stack_1_block1_MB_dw_swish/SigmoidSigmoid,stack_1_block1_MB_dw_bn/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????`?
stack_1_block1_MB_dw_swish/mulMul,stack_1_block1_MB_dw_bn/FusedBatchNormV3:y:0&stack_1_block1_MB_dw_swish/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
#stack_1_block1_MB_dw_swish/IdentityIdentity"stack_1_block1_MB_dw_swish/mul:z:0*
T0*/
_output_shapes
:?????????`?
$stack_1_block1_MB_dw_swish/IdentityN	IdentityN"stack_1_block1_MB_dw_swish/mul:z:0,stack_1_block1_MB_dw_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45717*J
_output_shapes8
6:?????????`:?????????`}
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_2/MeanMean-stack_1_block1_MB_dw_swish/IdentityN:output:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
.stack_1_block1_se_1_conv/Conv2D/ReadVariableOpReadVariableOp7stack_1_block1_se_1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
stack_1_block1_se_1_conv/Conv2DConv2D#tf.math.reduce_mean_2/Mean:output:06stack_1_block1_se_1_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_1_block1_se_1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_1_block1_se_1_conv/BiasAddBiasAdd(stack_1_block1_se_1_conv/Conv2D:output:07stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
stack_1_block1_se_swish/SigmoidSigmoid)stack_1_block1_se_1_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_1_block1_se_swish/mulMul)stack_1_block1_se_1_conv/BiasAdd:output:0#stack_1_block1_se_swish/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
 stack_1_block1_se_swish/IdentityIdentitystack_1_block1_se_swish/mul:z:0*
T0*/
_output_shapes
:??????????
!stack_1_block1_se_swish/IdentityN	IdentityNstack_1_block1_se_swish/mul:z:0)stack_1_block1_se_1_conv/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-45731*J
_output_shapes8
6:?????????:??????????
.stack_1_block1_se_2_conv/Conv2D/ReadVariableOpReadVariableOp7stack_1_block1_se_2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
stack_1_block1_se_2_conv/Conv2DConv2D*stack_1_block1_se_swish/IdentityN:output:06stack_1_block1_se_2_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_1_block1_se_2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_1_block1_se_2_conv/BiasAddBiasAdd(stack_1_block1_se_2_conv/Conv2D:output:07stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
!stack_1_block1_se_sigmoid/SigmoidSigmoid)stack_1_block1_se_2_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_1_block1_se_out/mulMul-stack_1_block1_MB_dw_swish/IdentityN:output:0%stack_1_block1_se_sigmoid/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOpReadVariableOp8stack_1_block1_mb_pw_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
 stack_1_block1_MB_pw_conv/Conv2DConv2Dstack_1_block1_se_out/mul:z:07stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
?
&stack_1_block1_MB_pw_bn/ReadVariableOpReadVariableOp/stack_1_block1_mb_pw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_1_block1_MB_pw_bn/ReadVariableOp_1ReadVariableOp1stack_1_block1_mb_pw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_1_block1_MB_pw_bn/FusedBatchNormV3FusedBatchNormV3)stack_1_block1_MB_pw_conv/Conv2D:output:0.stack_1_block1_MB_pw_bn/ReadVariableOp:value:00stack_1_block1_MB_pw_bn/ReadVariableOp_1:value:0?stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
&stack_1_block1_MB_pw_bn/AssignNewValueAssignVariableOp@stack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_resource5stack_1_block1_MB_pw_bn/FusedBatchNormV3:batch_mean:08^stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
(stack_1_block1_MB_pw_bn/AssignNewValue_1AssignVariableOpBstack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource9stack_1_block1_MB_pw_bn/FusedBatchNormV3:batch_variance:0:^stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0y
stack_1_block1_dropdrop/ShapeShape,stack_1_block1_MB_pw_bn/FusedBatchNormV3:y:0*
T0*
_output_shapes
:u
+stack_1_block1_dropdrop/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-stack_1_block1_dropdrop/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-stack_1_block1_dropdrop/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%stack_1_block1_dropdrop/strided_sliceStridedSlice&stack_1_block1_dropdrop/Shape:output:04stack_1_block1_dropdrop/strided_slice/stack:output:06stack_1_block1_dropdrop/strided_slice/stack_1:output:06stack_1_block1_dropdrop/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 stack_1_block1_dropdrop/packed/1Const*
_output_shapes
: *
dtype0*
value	B :b
 stack_1_block1_dropdrop/packed/2Const*
_output_shapes
: *
dtype0*
value	B :b
 stack_1_block1_dropdrop/packed/3Const*
_output_shapes
: *
dtype0*
value	B :?
stack_1_block1_dropdrop/packedPack.stack_1_block1_dropdrop/strided_slice:output:0)stack_1_block1_dropdrop/packed/1:output:0)stack_1_block1_dropdrop/packed/2:output:0)stack_1_block1_dropdrop/packed/3:output:0*
N*
T0*
_output_shapes
:j
%stack_1_block1_dropdrop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
#stack_1_block1_dropdrop/dropout/MulMul,stack_1_block1_MB_pw_bn/FusedBatchNormV3:y:0.stack_1_block1_dropdrop/dropout/Const:output:0*
T0*/
_output_shapes
:?????????`?
<stack_1_block1_dropdrop/dropout/random_uniform/RandomUniformRandomUniform'stack_1_block1_dropdrop/packed:output:0*
T0*/
_output_shapes
:?????????*
dtype0s
.stack_1_block1_dropdrop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??*>?
,stack_1_block1_dropdrop/dropout/GreaterEqualGreaterEqualEstack_1_block1_dropdrop/dropout/random_uniform/RandomUniform:output:07stack_1_block1_dropdrop/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:??????????
$stack_1_block1_dropdrop/dropout/CastCast0stack_1_block1_dropdrop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:??????????
%stack_1_block1_dropdrop/dropout/Mul_1Mul'stack_1_block1_dropdrop/dropout/Mul:z:0(stack_1_block1_dropdrop/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????`?
stack_1_block1_output/addAddV2,stack_1_block0_MB_pw_bn/FusedBatchNormV3:y:0)stack_1_block1_dropdrop/dropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????`?
post_conv/Conv2D/ReadVariableOpReadVariableOp(post_conv_conv2d_readvariableop_resource*'
_output_shapes
:?
*
dtype0?
post_conv/Conv2DConv2Dstack_1_block1_output/add:z:0'post_conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`?
*
paddingVALID*
strides
s
post_bn/ReadVariableOpReadVariableOppost_bn_readvariableop_resource*
_output_shapes	
:?
*
dtype0w
post_bn/ReadVariableOp_1ReadVariableOp!post_bn_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
'post_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp0post_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
)post_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp2post_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
post_bn/FusedBatchNormV3FusedBatchNormV3post_conv/Conv2D:output:0post_bn/ReadVariableOp:value:0 post_bn/ReadVariableOp_1:value:0/post_bn/FusedBatchNormV3/ReadVariableOp:value:01post_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????`?
:?
:?
:?
:?
:*
epsilon%o?:*
exponential_avg_factor%???=?
post_bn/AssignNewValueAssignVariableOp0post_bn_fusedbatchnormv3_readvariableop_resource%post_bn/FusedBatchNormV3:batch_mean:0(^post_bn/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
post_bn/AssignNewValue_1AssignVariableOp2post_bn_fusedbatchnormv3_readvariableop_1_resource)post_bn/FusedBatchNormV3:batch_variance:0*^post_bn/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0v
post_swish/SigmoidSigmoidpost_bn/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`?
?
post_swish/mulMulpost_bn/FusedBatchNormV3:y:0post_swish/Sigmoid:y:0*
T0*0
_output_shapes
:?????????`?
n
post_swish/IdentityIdentitypost_swish/mul:z:0*
T0*0
_output_shapes
:?????????`?
?
post_swish/IdentityN	IdentityNpost_swish/mul:z:0post_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45796*L
_output_shapes:
8:?????????`?
:?????????`?
p
avg_pool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
avg_pool/MeanMeanpost_swish/IdentityN:output:0(avg_pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????
\
head_drop/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?????
head_drop/dropout/MulMulavg_pool/Mean:output:0 head_drop/dropout/Const:output:0*
T0*(
_output_shapes
:??????????
]
head_drop/dropout/ShapeShapeavg_pool/Mean:output:0*
T0*
_output_shapes
:?
.head_drop/dropout/random_uniform/RandomUniformRandomUniform head_drop/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????
*
dtype0e
 head_drop/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
head_drop/dropout/GreaterEqualGreaterEqual7head_drop/dropout/random_uniform/RandomUniform:output:0)head_drop/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????
?
head_drop/dropout/CastCast"head_drop/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????
?
head_drop/dropout/Mul_1Mulhead_drop/dropout/Mul:z:0head_drop/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????
?
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
predictions/MatMulMatMulhead_drop/dropout/Mul_1:z:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
predictions/SoftmaxSoftmaxpredictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
IdentityIdentitypredictions/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^post_bn/AssignNewValue^post_bn/AssignNewValue_1(^post_bn/FusedBatchNormV3/ReadVariableOp*^post_bn/FusedBatchNormV3/ReadVariableOp_1^post_bn/ReadVariableOp^post_bn/ReadVariableOp_1 ^post_conv/Conv2D/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp/^stack_0_block0_MB_dw_/depthwise/ReadVariableOp'^stack_0_block0_MB_dw_bn/AssignNewValue)^stack_0_block0_MB_dw_bn/AssignNewValue_18^stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:^stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_0_block0_MB_dw_bn/ReadVariableOp)^stack_0_block0_MB_dw_bn/ReadVariableOp_1'^stack_0_block0_MB_pw_bn/AssignNewValue)^stack_0_block0_MB_pw_bn/AssignNewValue_18^stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:^stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_0_block0_MB_pw_bn/ReadVariableOp)^stack_0_block0_MB_pw_bn/ReadVariableOp_10^stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp0^stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp/^stack_0_block0_se_1_conv/Conv2D/ReadVariableOp0^stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp/^stack_0_block0_se_2_conv/Conv2D/ReadVariableOp/^stack_1_block0_MB_dw_/depthwise/ReadVariableOp'^stack_1_block0_MB_dw_bn/AssignNewValue)^stack_1_block0_MB_dw_bn/AssignNewValue_18^stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:^stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_1_block0_MB_dw_bn/ReadVariableOp)^stack_1_block0_MB_dw_bn/ReadVariableOp_1'^stack_1_block0_MB_pw_bn/AssignNewValue)^stack_1_block0_MB_pw_bn/AssignNewValue_18^stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:^stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_1_block0_MB_pw_bn/ReadVariableOp)^stack_1_block0_MB_pw_bn/ReadVariableOp_10^stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp0^stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp/^stack_1_block0_se_1_conv/Conv2D/ReadVariableOp0^stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp/^stack_1_block0_se_2_conv/Conv2D/ReadVariableOp/^stack_1_block1_MB_dw_/depthwise/ReadVariableOp'^stack_1_block1_MB_dw_bn/AssignNewValue)^stack_1_block1_MB_dw_bn/AssignNewValue_18^stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:^stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_1_block1_MB_dw_bn/ReadVariableOp)^stack_1_block1_MB_dw_bn/ReadVariableOp_1'^stack_1_block1_MB_pw_bn/AssignNewValue)^stack_1_block1_MB_pw_bn/AssignNewValue_18^stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:^stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_1_block1_MB_pw_bn/ReadVariableOp)^stack_1_block1_MB_pw_bn/ReadVariableOp_10^stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp0^stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp/^stack_1_block1_se_1_conv/Conv2D/ReadVariableOp0^stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp/^stack_1_block1_se_2_conv/Conv2D/ReadVariableOp^stem_bn/AssignNewValue^stem_bn/AssignNewValue_1(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 20
post_bn/AssignNewValuepost_bn/AssignNewValue24
post_bn/AssignNewValue_1post_bn/AssignNewValue_12R
'post_bn/FusedBatchNormV3/ReadVariableOp'post_bn/FusedBatchNormV3/ReadVariableOp2V
)post_bn/FusedBatchNormV3/ReadVariableOp_1)post_bn/FusedBatchNormV3/ReadVariableOp_120
post_bn/ReadVariableOppost_bn/ReadVariableOp24
post_bn/ReadVariableOp_1post_bn/ReadVariableOp_12B
post_conv/Conv2D/ReadVariableOppost_conv/Conv2D/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp2`
.stack_0_block0_MB_dw_/depthwise/ReadVariableOp.stack_0_block0_MB_dw_/depthwise/ReadVariableOp2P
&stack_0_block0_MB_dw_bn/AssignNewValue&stack_0_block0_MB_dw_bn/AssignNewValue2T
(stack_0_block0_MB_dw_bn/AssignNewValue_1(stack_0_block0_MB_dw_bn/AssignNewValue_12r
7stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp7stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_19stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_0_block0_MB_dw_bn/ReadVariableOp&stack_0_block0_MB_dw_bn/ReadVariableOp2T
(stack_0_block0_MB_dw_bn/ReadVariableOp_1(stack_0_block0_MB_dw_bn/ReadVariableOp_12P
&stack_0_block0_MB_pw_bn/AssignNewValue&stack_0_block0_MB_pw_bn/AssignNewValue2T
(stack_0_block0_MB_pw_bn/AssignNewValue_1(stack_0_block0_MB_pw_bn/AssignNewValue_12r
7stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp7stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_19stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_0_block0_MB_pw_bn/ReadVariableOp&stack_0_block0_MB_pw_bn/ReadVariableOp2T
(stack_0_block0_MB_pw_bn/ReadVariableOp_1(stack_0_block0_MB_pw_bn/ReadVariableOp_12b
/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp2b
/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp2`
.stack_0_block0_se_1_conv/Conv2D/ReadVariableOp.stack_0_block0_se_1_conv/Conv2D/ReadVariableOp2b
/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp2`
.stack_0_block0_se_2_conv/Conv2D/ReadVariableOp.stack_0_block0_se_2_conv/Conv2D/ReadVariableOp2`
.stack_1_block0_MB_dw_/depthwise/ReadVariableOp.stack_1_block0_MB_dw_/depthwise/ReadVariableOp2P
&stack_1_block0_MB_dw_bn/AssignNewValue&stack_1_block0_MB_dw_bn/AssignNewValue2T
(stack_1_block0_MB_dw_bn/AssignNewValue_1(stack_1_block0_MB_dw_bn/AssignNewValue_12r
7stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp7stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_19stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_1_block0_MB_dw_bn/ReadVariableOp&stack_1_block0_MB_dw_bn/ReadVariableOp2T
(stack_1_block0_MB_dw_bn/ReadVariableOp_1(stack_1_block0_MB_dw_bn/ReadVariableOp_12P
&stack_1_block0_MB_pw_bn/AssignNewValue&stack_1_block0_MB_pw_bn/AssignNewValue2T
(stack_1_block0_MB_pw_bn/AssignNewValue_1(stack_1_block0_MB_pw_bn/AssignNewValue_12r
7stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp7stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_19stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_1_block0_MB_pw_bn/ReadVariableOp&stack_1_block0_MB_pw_bn/ReadVariableOp2T
(stack_1_block0_MB_pw_bn/ReadVariableOp_1(stack_1_block0_MB_pw_bn/ReadVariableOp_12b
/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp2b
/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp2`
.stack_1_block0_se_1_conv/Conv2D/ReadVariableOp.stack_1_block0_se_1_conv/Conv2D/ReadVariableOp2b
/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp2`
.stack_1_block0_se_2_conv/Conv2D/ReadVariableOp.stack_1_block0_se_2_conv/Conv2D/ReadVariableOp2`
.stack_1_block1_MB_dw_/depthwise/ReadVariableOp.stack_1_block1_MB_dw_/depthwise/ReadVariableOp2P
&stack_1_block1_MB_dw_bn/AssignNewValue&stack_1_block1_MB_dw_bn/AssignNewValue2T
(stack_1_block1_MB_dw_bn/AssignNewValue_1(stack_1_block1_MB_dw_bn/AssignNewValue_12r
7stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp7stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_19stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_1_block1_MB_dw_bn/ReadVariableOp&stack_1_block1_MB_dw_bn/ReadVariableOp2T
(stack_1_block1_MB_dw_bn/ReadVariableOp_1(stack_1_block1_MB_dw_bn/ReadVariableOp_12P
&stack_1_block1_MB_pw_bn/AssignNewValue&stack_1_block1_MB_pw_bn/AssignNewValue2T
(stack_1_block1_MB_pw_bn/AssignNewValue_1(stack_1_block1_MB_pw_bn/AssignNewValue_12r
7stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp7stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_19stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_1_block1_MB_pw_bn/ReadVariableOp&stack_1_block1_MB_pw_bn/ReadVariableOp2T
(stack_1_block1_MB_pw_bn/ReadVariableOp_1(stack_1_block1_MB_pw_bn/ReadVariableOp_12b
/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp2b
/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp2`
.stack_1_block1_se_1_conv/Conv2D/ReadVariableOp.stack_1_block1_se_1_conv/Conv2D/ReadVariableOp2b
/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp2`
.stack_1_block1_se_2_conv/Conv2D/ReadVariableOp.stack_1_block1_se_2_conv/Conv2D/ReadVariableOp20
stem_bn/AssignNewValuestem_bn/AssignNewValue24
stem_bn/AssignNewValue_1stem_bn/AssignNewValue_12R
'stem_bn/FusedBatchNormV3/ReadVariableOp'stem_bn/FusedBatchNormV3/ReadVariableOp2V
)stem_bn/FusedBatchNormV3/ReadVariableOp_1)stem_bn/FusedBatchNormV3/ReadVariableOp_120
stem_bn/ReadVariableOpstem_bn/ReadVariableOp24
stem_bn/ReadVariableOp_1stem_bn/ReadVariableOp_12B
stem_conv/Conv2D/ReadVariableOpstem_conv/Conv2D/ReadVariableOp:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?

?
S__inference_stack_1_block0_se_1_conv_layer_call_and_return_conditional_losses_46524

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46813

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_45064
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: #
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: 

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:$

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:$

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:$

unknown_37:

unknown_38:$

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:%

unknown_46:?


unknown_47:	?


unknown_48:	?


unknown_49:	?


unknown_50:	?


unknown_51:	?


unknown_52:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_42215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:????????? ?
!
_user_specified_name	input_1
?
?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43095

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

?
F__inference_predictions_layer_call_and_return_conditional_losses_47364

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
a
5__inference_stack_1_block1_output_layer_call_fn_47136
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_output_layer_call_and_return_conditional_losses_43275h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????`:?????????`:Y U
/
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????`
"
_user_specified_name
inputs/1
?
?
"__inference_internal_grad_fn_47721
result_grads_0
result_grads_19
5sigmoid_efficientnet_stack_1_block0_se_1_conv_biasadd
identity?
SigmoidSigmoid5sigmoid_efficientnet_stack_1_block0_se_1_conv_biasadd^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:??????????
mulMul5sigmoid_efficientnet_stack_1_block0_se_1_conv_biasaddsub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46831

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?d
!__inference__traced_restore_48831
file_prefix;
!assignvariableop_stem_conv_kernel: .
 assignvariableop_1_stem_bn_gamma: -
assignvariableop_2_stem_bn_beta: 4
&assignvariableop_3_stem_bn_moving_mean: 8
*assignvariableop_4_stem_bn_moving_variance: S
9assignvariableop_5_stack_0_block0_mb_dw__depthwise_kernel: >
0assignvariableop_6_stack_0_block0_mb_dw_bn_gamma: =
/assignvariableop_7_stack_0_block0_mb_dw_bn_beta: D
6assignvariableop_8_stack_0_block0_mb_dw_bn_moving_mean: H
:assignvariableop_9_stack_0_block0_mb_dw_bn_moving_variance: M
3assignvariableop_10_stack_0_block0_se_1_conv_kernel: ?
1assignvariableop_11_stack_0_block0_se_1_conv_bias:M
3assignvariableop_12_stack_0_block0_se_2_conv_kernel: ?
1assignvariableop_13_stack_0_block0_se_2_conv_bias: N
4assignvariableop_14_stack_0_block0_mb_pw_conv_kernel: ?
1assignvariableop_15_stack_0_block0_mb_pw_bn_gamma:>
0assignvariableop_16_stack_0_block0_mb_pw_bn_beta:E
7assignvariableop_17_stack_0_block0_mb_pw_bn_moving_mean:I
;assignvariableop_18_stack_0_block0_mb_pw_bn_moving_variance:T
:assignvariableop_19_stack_1_block0_mb_dw__depthwise_kernel:?
1assignvariableop_20_stack_1_block0_mb_dw_bn_gamma:>
0assignvariableop_21_stack_1_block0_mb_dw_bn_beta:E
7assignvariableop_22_stack_1_block0_mb_dw_bn_moving_mean:I
;assignvariableop_23_stack_1_block0_mb_dw_bn_moving_variance:M
3assignvariableop_24_stack_1_block0_se_1_conv_kernel:?
1assignvariableop_25_stack_1_block0_se_1_conv_bias:M
3assignvariableop_26_stack_1_block0_se_2_conv_kernel:?
1assignvariableop_27_stack_1_block0_se_2_conv_bias:N
4assignvariableop_28_stack_1_block0_mb_pw_conv_kernel:?
1assignvariableop_29_stack_1_block0_mb_pw_bn_gamma:>
0assignvariableop_30_stack_1_block0_mb_pw_bn_beta:E
7assignvariableop_31_stack_1_block0_mb_pw_bn_moving_mean:I
;assignvariableop_32_stack_1_block0_mb_pw_bn_moving_variance:T
:assignvariableop_33_stack_1_block1_mb_dw__depthwise_kernel:?
1assignvariableop_34_stack_1_block1_mb_dw_bn_gamma:>
0assignvariableop_35_stack_1_block1_mb_dw_bn_beta:E
7assignvariableop_36_stack_1_block1_mb_dw_bn_moving_mean:I
;assignvariableop_37_stack_1_block1_mb_dw_bn_moving_variance:M
3assignvariableop_38_stack_1_block1_se_1_conv_kernel:?
1assignvariableop_39_stack_1_block1_se_1_conv_bias:M
3assignvariableop_40_stack_1_block1_se_2_conv_kernel:?
1assignvariableop_41_stack_1_block1_se_2_conv_bias:N
4assignvariableop_42_stack_1_block1_mb_pw_conv_kernel:?
1assignvariableop_43_stack_1_block1_mb_pw_bn_gamma:>
0assignvariableop_44_stack_1_block1_mb_pw_bn_beta:E
7assignvariableop_45_stack_1_block1_mb_pw_bn_moving_mean:I
;assignvariableop_46_stack_1_block1_mb_pw_bn_moving_variance:?
$assignvariableop_47_post_conv_kernel:?
0
!assignvariableop_48_post_bn_gamma:	?
/
 assignvariableop_49_post_bn_beta:	?
6
'assignvariableop_50_post_bn_moving_mean:	?
:
+assignvariableop_51_post_bn_moving_variance:	?
9
&assignvariableop_52_predictions_kernel:	?
2
$assignvariableop_53_predictions_bias:$
assignvariableop_54_beta_1: $
assignvariableop_55_beta_2: #
assignvariableop_56_decay: +
!assignvariableop_57_learning_rate: '
assignvariableop_58_adam_iter:	 #
assignvariableop_59_total: #
assignvariableop_60_count: %
assignvariableop_61_total_1: %
assignvariableop_62_count_1: E
+assignvariableop_63_adam_stem_conv_kernel_m: 6
(assignvariableop_64_adam_stem_bn_gamma_m: 5
'assignvariableop_65_adam_stem_bn_beta_m: [
Aassignvariableop_66_adam_stack_0_block0_mb_dw__depthwise_kernel_m: F
8assignvariableop_67_adam_stack_0_block0_mb_dw_bn_gamma_m: E
7assignvariableop_68_adam_stack_0_block0_mb_dw_bn_beta_m: T
:assignvariableop_69_adam_stack_0_block0_se_1_conv_kernel_m: F
8assignvariableop_70_adam_stack_0_block0_se_1_conv_bias_m:T
:assignvariableop_71_adam_stack_0_block0_se_2_conv_kernel_m: F
8assignvariableop_72_adam_stack_0_block0_se_2_conv_bias_m: U
;assignvariableop_73_adam_stack_0_block0_mb_pw_conv_kernel_m: F
8assignvariableop_74_adam_stack_0_block0_mb_pw_bn_gamma_m:E
7assignvariableop_75_adam_stack_0_block0_mb_pw_bn_beta_m:[
Aassignvariableop_76_adam_stack_1_block0_mb_dw__depthwise_kernel_m:F
8assignvariableop_77_adam_stack_1_block0_mb_dw_bn_gamma_m:E
7assignvariableop_78_adam_stack_1_block0_mb_dw_bn_beta_m:T
:assignvariableop_79_adam_stack_1_block0_se_1_conv_kernel_m:F
8assignvariableop_80_adam_stack_1_block0_se_1_conv_bias_m:T
:assignvariableop_81_adam_stack_1_block0_se_2_conv_kernel_m:F
8assignvariableop_82_adam_stack_1_block0_se_2_conv_bias_m:U
;assignvariableop_83_adam_stack_1_block0_mb_pw_conv_kernel_m:F
8assignvariableop_84_adam_stack_1_block0_mb_pw_bn_gamma_m:E
7assignvariableop_85_adam_stack_1_block0_mb_pw_bn_beta_m:[
Aassignvariableop_86_adam_stack_1_block1_mb_dw__depthwise_kernel_m:F
8assignvariableop_87_adam_stack_1_block1_mb_dw_bn_gamma_m:E
7assignvariableop_88_adam_stack_1_block1_mb_dw_bn_beta_m:T
:assignvariableop_89_adam_stack_1_block1_se_1_conv_kernel_m:F
8assignvariableop_90_adam_stack_1_block1_se_1_conv_bias_m:T
:assignvariableop_91_adam_stack_1_block1_se_2_conv_kernel_m:F
8assignvariableop_92_adam_stack_1_block1_se_2_conv_bias_m:U
;assignvariableop_93_adam_stack_1_block1_mb_pw_conv_kernel_m:F
8assignvariableop_94_adam_stack_1_block1_mb_pw_bn_gamma_m:E
7assignvariableop_95_adam_stack_1_block1_mb_pw_bn_beta_m:F
+assignvariableop_96_adam_post_conv_kernel_m:?
7
(assignvariableop_97_adam_post_bn_gamma_m:	?
6
'assignvariableop_98_adam_post_bn_beta_m:	?
@
-assignvariableop_99_adam_predictions_kernel_m:	?
:
,assignvariableop_100_adam_predictions_bias_m:F
,assignvariableop_101_adam_stem_conv_kernel_v: 7
)assignvariableop_102_adam_stem_bn_gamma_v: 6
(assignvariableop_103_adam_stem_bn_beta_v: \
Bassignvariableop_104_adam_stack_0_block0_mb_dw__depthwise_kernel_v: G
9assignvariableop_105_adam_stack_0_block0_mb_dw_bn_gamma_v: F
8assignvariableop_106_adam_stack_0_block0_mb_dw_bn_beta_v: U
;assignvariableop_107_adam_stack_0_block0_se_1_conv_kernel_v: G
9assignvariableop_108_adam_stack_0_block0_se_1_conv_bias_v:U
;assignvariableop_109_adam_stack_0_block0_se_2_conv_kernel_v: G
9assignvariableop_110_adam_stack_0_block0_se_2_conv_bias_v: V
<assignvariableop_111_adam_stack_0_block0_mb_pw_conv_kernel_v: G
9assignvariableop_112_adam_stack_0_block0_mb_pw_bn_gamma_v:F
8assignvariableop_113_adam_stack_0_block0_mb_pw_bn_beta_v:\
Bassignvariableop_114_adam_stack_1_block0_mb_dw__depthwise_kernel_v:G
9assignvariableop_115_adam_stack_1_block0_mb_dw_bn_gamma_v:F
8assignvariableop_116_adam_stack_1_block0_mb_dw_bn_beta_v:U
;assignvariableop_117_adam_stack_1_block0_se_1_conv_kernel_v:G
9assignvariableop_118_adam_stack_1_block0_se_1_conv_bias_v:U
;assignvariableop_119_adam_stack_1_block0_se_2_conv_kernel_v:G
9assignvariableop_120_adam_stack_1_block0_se_2_conv_bias_v:V
<assignvariableop_121_adam_stack_1_block0_mb_pw_conv_kernel_v:G
9assignvariableop_122_adam_stack_1_block0_mb_pw_bn_gamma_v:F
8assignvariableop_123_adam_stack_1_block0_mb_pw_bn_beta_v:\
Bassignvariableop_124_adam_stack_1_block1_mb_dw__depthwise_kernel_v:G
9assignvariableop_125_adam_stack_1_block1_mb_dw_bn_gamma_v:F
8assignvariableop_126_adam_stack_1_block1_mb_dw_bn_beta_v:U
;assignvariableop_127_adam_stack_1_block1_se_1_conv_kernel_v:G
9assignvariableop_128_adam_stack_1_block1_se_1_conv_bias_v:U
;assignvariableop_129_adam_stack_1_block1_se_2_conv_kernel_v:G
9assignvariableop_130_adam_stack_1_block1_se_2_conv_bias_v:V
<assignvariableop_131_adam_stack_1_block1_mb_pw_conv_kernel_v:G
9assignvariableop_132_adam_stack_1_block1_mb_pw_bn_gamma_v:F
8assignvariableop_133_adam_stack_1_block1_mb_pw_bn_beta_v:G
,assignvariableop_134_adam_post_conv_kernel_v:?
8
)assignvariableop_135_adam_post_bn_gamma_v:	?
7
(assignvariableop_136_adam_post_bn_beta_v:	?
A
.assignvariableop_137_adam_predictions_kernel_v:	?
:
,assignvariableop_138_adam_predictions_bias_v:
identity_140??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?O
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?N
value?NB?N?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-19/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-19/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_stem_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_stem_bn_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_stem_bn_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_stem_bn_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_stem_bn_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_stack_0_block0_mb_dw__depthwise_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp0assignvariableop_6_stack_0_block0_mb_dw_bn_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_stack_0_block0_mb_dw_bn_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_stack_0_block0_mb_dw_bn_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp:assignvariableop_9_stack_0_block0_mb_dw_bn_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp3assignvariableop_10_stack_0_block0_se_1_conv_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp1assignvariableop_11_stack_0_block0_se_1_conv_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp3assignvariableop_12_stack_0_block0_se_2_conv_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp1assignvariableop_13_stack_0_block0_se_2_conv_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp4assignvariableop_14_stack_0_block0_mb_pw_conv_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp1assignvariableop_15_stack_0_block0_mb_pw_bn_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp0assignvariableop_16_stack_0_block0_mb_pw_bn_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp7assignvariableop_17_stack_0_block0_mb_pw_bn_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp;assignvariableop_18_stack_0_block0_mb_pw_bn_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_stack_1_block0_mb_dw__depthwise_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_stack_1_block0_mb_dw_bn_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp0assignvariableop_21_stack_1_block0_mb_dw_bn_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp7assignvariableop_22_stack_1_block0_mb_dw_bn_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp;assignvariableop_23_stack_1_block0_mb_dw_bn_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp3assignvariableop_24_stack_1_block0_se_1_conv_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp1assignvariableop_25_stack_1_block0_se_1_conv_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp3assignvariableop_26_stack_1_block0_se_2_conv_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp1assignvariableop_27_stack_1_block0_se_2_conv_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_stack_1_block0_mb_pw_conv_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_stack_1_block0_mb_pw_bn_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp0assignvariableop_30_stack_1_block0_mb_pw_bn_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp7assignvariableop_31_stack_1_block0_mb_pw_bn_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp;assignvariableop_32_stack_1_block0_mb_pw_bn_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp:assignvariableop_33_stack_1_block1_mb_dw__depthwise_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp1assignvariableop_34_stack_1_block1_mb_dw_bn_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp0assignvariableop_35_stack_1_block1_mb_dw_bn_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp7assignvariableop_36_stack_1_block1_mb_dw_bn_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp;assignvariableop_37_stack_1_block1_mb_dw_bn_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_stack_1_block1_se_1_conv_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp1assignvariableop_39_stack_1_block1_se_1_conv_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp3assignvariableop_40_stack_1_block1_se_2_conv_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp1assignvariableop_41_stack_1_block1_se_2_conv_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp4assignvariableop_42_stack_1_block1_mb_pw_conv_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp1assignvariableop_43_stack_1_block1_mb_pw_bn_gammaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp0assignvariableop_44_stack_1_block1_mb_pw_bn_betaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp7assignvariableop_45_stack_1_block1_mb_pw_bn_moving_meanIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp;assignvariableop_46_stack_1_block1_mb_pw_bn_moving_varianceIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp$assignvariableop_47_post_conv_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp!assignvariableop_48_post_bn_gammaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp assignvariableop_49_post_bn_betaIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp'assignvariableop_50_post_bn_moving_meanIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_post_bn_moving_varianceIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp&assignvariableop_52_predictions_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp$assignvariableop_53_predictions_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpassignvariableop_54_beta_1Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpassignvariableop_55_beta_2Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpassignvariableop_56_decayIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp!assignvariableop_57_learning_rateIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_58AssignVariableOpassignvariableop_58_adam_iterIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOpassignvariableop_59_totalIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOpassignvariableop_60_countIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOpassignvariableop_61_total_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOpassignvariableop_62_count_1Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_stem_conv_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_stem_bn_gamma_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp'assignvariableop_65_adam_stem_bn_beta_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOpAassignvariableop_66_adam_stack_0_block0_mb_dw__depthwise_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp8assignvariableop_67_adam_stack_0_block0_mb_dw_bn_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp7assignvariableop_68_adam_stack_0_block0_mb_dw_bn_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp:assignvariableop_69_adam_stack_0_block0_se_1_conv_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp8assignvariableop_70_adam_stack_0_block0_se_1_conv_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp:assignvariableop_71_adam_stack_0_block0_se_2_conv_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp8assignvariableop_72_adam_stack_0_block0_se_2_conv_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp;assignvariableop_73_adam_stack_0_block0_mb_pw_conv_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp8assignvariableop_74_adam_stack_0_block0_mb_pw_bn_gamma_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_stack_0_block0_mb_pw_bn_beta_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOpAassignvariableop_76_adam_stack_1_block0_mb_dw__depthwise_kernel_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp8assignvariableop_77_adam_stack_1_block0_mb_dw_bn_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp7assignvariableop_78_adam_stack_1_block0_mb_dw_bn_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp:assignvariableop_79_adam_stack_1_block0_se_1_conv_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp8assignvariableop_80_adam_stack_1_block0_se_1_conv_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp:assignvariableop_81_adam_stack_1_block0_se_2_conv_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp8assignvariableop_82_adam_stack_1_block0_se_2_conv_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp;assignvariableop_83_adam_stack_1_block0_mb_pw_conv_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp8assignvariableop_84_adam_stack_1_block0_mb_pw_bn_gamma_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp7assignvariableop_85_adam_stack_1_block0_mb_pw_bn_beta_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOpAassignvariableop_86_adam_stack_1_block1_mb_dw__depthwise_kernel_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp8assignvariableop_87_adam_stack_1_block1_mb_dw_bn_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp7assignvariableop_88_adam_stack_1_block1_mb_dw_bn_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp:assignvariableop_89_adam_stack_1_block1_se_1_conv_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp8assignvariableop_90_adam_stack_1_block1_se_1_conv_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp:assignvariableop_91_adam_stack_1_block1_se_2_conv_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp8assignvariableop_92_adam_stack_1_block1_se_2_conv_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp;assignvariableop_93_adam_stack_1_block1_mb_pw_conv_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp8assignvariableop_94_adam_stack_1_block1_mb_pw_bn_gamma_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp7assignvariableop_95_adam_stack_1_block1_mb_pw_bn_beta_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp+assignvariableop_96_adam_post_conv_kernel_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp(assignvariableop_97_adam_post_bn_gamma_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp'assignvariableop_98_adam_post_bn_beta_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOp-assignvariableop_99_adam_predictions_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOp,assignvariableop_100_adam_predictions_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_stem_conv_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOp)assignvariableop_102_adam_stem_bn_gamma_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp(assignvariableop_103_adam_stem_bn_beta_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOpBassignvariableop_104_adam_stack_0_block0_mb_dw__depthwise_kernel_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOp9assignvariableop_105_adam_stack_0_block0_mb_dw_bn_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp8assignvariableop_106_adam_stack_0_block0_mb_dw_bn_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOp;assignvariableop_107_adam_stack_0_block0_se_1_conv_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOp9assignvariableop_108_adam_stack_0_block0_se_1_conv_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp;assignvariableop_109_adam_stack_0_block0_se_2_conv_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOp9assignvariableop_110_adam_stack_0_block0_se_2_conv_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOp<assignvariableop_111_adam_stack_0_block0_mb_pw_conv_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOp9assignvariableop_112_adam_stack_0_block0_mb_pw_bn_gamma_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOp8assignvariableop_113_adam_stack_0_block0_mb_pw_bn_beta_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOpBassignvariableop_114_adam_stack_1_block0_mb_dw__depthwise_kernel_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOp9assignvariableop_115_adam_stack_1_block0_mb_dw_bn_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOp8assignvariableop_116_adam_stack_1_block0_mb_dw_bn_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOp;assignvariableop_117_adam_stack_1_block0_se_1_conv_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOp9assignvariableop_118_adam_stack_1_block0_se_1_conv_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOp;assignvariableop_119_adam_stack_1_block0_se_2_conv_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOp9assignvariableop_120_adam_stack_1_block0_se_2_conv_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOp<assignvariableop_121_adam_stack_1_block0_mb_pw_conv_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOp9assignvariableop_122_adam_stack_1_block0_mb_pw_bn_gamma_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_123AssignVariableOp8assignvariableop_123_adam_stack_1_block0_mb_pw_bn_beta_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOpBassignvariableop_124_adam_stack_1_block1_mb_dw__depthwise_kernel_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_125AssignVariableOp9assignvariableop_125_adam_stack_1_block1_mb_dw_bn_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_126AssignVariableOp8assignvariableop_126_adam_stack_1_block1_mb_dw_bn_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_127AssignVariableOp;assignvariableop_127_adam_stack_1_block1_se_1_conv_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_128AssignVariableOp9assignvariableop_128_adam_stack_1_block1_se_1_conv_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_129AssignVariableOp;assignvariableop_129_adam_stack_1_block1_se_2_conv_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_130AssignVariableOp9assignvariableop_130_adam_stack_1_block1_se_2_conv_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_131AssignVariableOp<assignvariableop_131_adam_stack_1_block1_mb_pw_conv_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_132AssignVariableOp9assignvariableop_132_adam_stack_1_block1_mb_pw_bn_gamma_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_133AssignVariableOp8assignvariableop_133_adam_stack_1_block1_mb_pw_bn_beta_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_134AssignVariableOp,assignvariableop_134_adam_post_conv_kernel_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_135AssignVariableOp)assignvariableop_135_adam_post_bn_gamma_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_136AssignVariableOp(assignvariableop_136_adam_post_bn_beta_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_137AssignVariableOp.assignvariableop_137_adam_predictions_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_138AssignVariableOp,assignvariableop_138_adam_predictions_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_139Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_140IdentityIdentity_139:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_140Identity_140:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
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
AssignVariableOp_5AssignVariableOp_52*
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
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

n
"__inference_internal_grad_fn_48186
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?	
?
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46607

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42493?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_stem_bn_layer_call_and_return_conditional_losses_42268

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47961
result_grads_0
result_grads_1,
(sigmoid_stack_1_block0_se_1_conv_biasadd
identity?
SigmoidSigmoid(sigmoid_stack_1_block0_se_1_conv_biasadd^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????w
mulMul(sigmoid_stack_1_block0_se_1_conv_biasaddsub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?	
?
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46243

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42396?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46323

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47059

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
G__inference_EfficientNet_layer_call_and_return_conditional_losses_44782
input_1)
stem_conv_44624: 
stem_bn_44627: 
stem_bn_44629: 
stem_bn_44631: 
stem_bn_44633: 5
stack_0_block0_mb_dw__44637: +
stack_0_block0_mb_dw_bn_44640: +
stack_0_block0_mb_dw_bn_44642: +
stack_0_block0_mb_dw_bn_44644: +
stack_0_block0_mb_dw_bn_44646: 8
stack_0_block0_se_1_conv_44652: ,
stack_0_block0_se_1_conv_44654:8
stack_0_block0_se_2_conv_44658: ,
stack_0_block0_se_2_conv_44660: 9
stack_0_block0_mb_pw_conv_44665: +
stack_0_block0_mb_pw_bn_44668:+
stack_0_block0_mb_pw_bn_44670:+
stack_0_block0_mb_pw_bn_44672:+
stack_0_block0_mb_pw_bn_44674:5
stack_1_block0_mb_dw__44678:+
stack_1_block0_mb_dw_bn_44681:+
stack_1_block0_mb_dw_bn_44683:+
stack_1_block0_mb_dw_bn_44685:+
stack_1_block0_mb_dw_bn_44687:8
stack_1_block0_se_1_conv_44693:,
stack_1_block0_se_1_conv_44695:8
stack_1_block0_se_2_conv_44699:,
stack_1_block0_se_2_conv_44701:9
stack_1_block0_mb_pw_conv_44706:+
stack_1_block0_mb_pw_bn_44709:+
stack_1_block0_mb_pw_bn_44711:+
stack_1_block0_mb_pw_bn_44713:+
stack_1_block0_mb_pw_bn_44715:5
stack_1_block1_mb_dw__44719:+
stack_1_block1_mb_dw_bn_44722:+
stack_1_block1_mb_dw_bn_44724:+
stack_1_block1_mb_dw_bn_44726:+
stack_1_block1_mb_dw_bn_44728:8
stack_1_block1_se_1_conv_44734:,
stack_1_block1_se_1_conv_44736:8
stack_1_block1_se_2_conv_44740:,
stack_1_block1_se_2_conv_44742:9
stack_1_block1_mb_pw_conv_44747:+
stack_1_block1_mb_pw_bn_44750:+
stack_1_block1_mb_pw_bn_44752:+
stack_1_block1_mb_pw_bn_44754:+
stack_1_block1_mb_pw_bn_44756:*
post_conv_44761:?

post_bn_44764:	?

post_bn_44766:	?

post_bn_44768:	?

post_bn_44770:	?
$
predictions_44776:	?

predictions_44778:
identity??post_bn/StatefulPartitionedCall?!post_conv/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?-stack_0_block0_MB_dw_/StatefulPartitionedCall?/stack_0_block0_MB_dw_bn/StatefulPartitionedCall?/stack_0_block0_MB_pw_bn/StatefulPartitionedCall?1stack_0_block0_MB_pw_conv/StatefulPartitionedCall?0stack_0_block0_se_1_conv/StatefulPartitionedCall?0stack_0_block0_se_2_conv/StatefulPartitionedCall?-stack_1_block0_MB_dw_/StatefulPartitionedCall?/stack_1_block0_MB_dw_bn/StatefulPartitionedCall?/stack_1_block0_MB_pw_bn/StatefulPartitionedCall?1stack_1_block0_MB_pw_conv/StatefulPartitionedCall?0stack_1_block0_se_1_conv/StatefulPartitionedCall?0stack_1_block0_se_2_conv/StatefulPartitionedCall?-stack_1_block1_MB_dw_/StatefulPartitionedCall?/stack_1_block1_MB_dw_bn/StatefulPartitionedCall?/stack_1_block1_MB_pw_bn/StatefulPartitionedCall?1stack_1_block1_MB_pw_conv/StatefulPartitionedCall?0stack_1_block1_se_1_conv/StatefulPartitionedCall?0stack_1_block1_se_2_conv/StatefulPartitionedCall?stem_bn/StatefulPartitionedCall?!stem_conv/StatefulPartitionedCall?
!stem_conv/StatefulPartitionedCallStatefulPartitionedCallinput_1stem_conv_44624*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_42754?
stem_bn/StatefulPartitionedCallStatefulPartitionedCall*stem_conv/StatefulPartitionedCall:output:0stem_bn_44627stem_bn_44629stem_bn_44631stem_bn_44633*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_42775?
stem_swish/PartitionedCallPartitionedCall(stem_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_stem_swish_layer_call_and_return_conditional_losses_42795?
-stack_0_block0_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall#stem_swish/PartitionedCall:output:0stack_0_block0_mb_dw__44637*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_MB_dw__layer_call_and_return_conditional_losses_42806?
/stack_0_block0_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_0_block0_MB_dw_/StatefulPartitionedCall:output:0stack_0_block0_mb_dw_bn_44640stack_0_block0_mb_dw_bn_44642stack_0_block0_mb_dw_bn_44644stack_0_block0_mb_dw_bn_44646*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42827?
*stack_0_block0_MB_dw_swish/PartitionedCallPartitionedCall8stack_0_block0_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_42847{
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean/MeanMean3stack_0_block0_MB_dw_swish/PartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(?
0stack_0_block0_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_mean/Mean:output:0stack_0_block0_se_1_conv_44652stack_0_block0_se_1_conv_44654*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_1_conv_layer_call_and_return_conditional_losses_42861?
'stack_0_block0_se_swish/PartitionedCallPartitionedCall9stack_0_block0_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_42877?
0stack_0_block0_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_0_block0_se_swish/PartitionedCall:output:0stack_0_block0_se_2_conv_44658stack_0_block0_se_2_conv_44660*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_2_conv_layer_call_and_return_conditional_losses_42889?
)stack_0_block0_se_sigmoid/PartitionedCallPartitionedCall9stack_0_block0_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_se_sigmoid_layer_call_and_return_conditional_losses_42900?
%stack_0_block0_se_out/PartitionedCallPartitionedCall3stack_0_block0_MB_dw_swish/PartitionedCall:output:02stack_0_block0_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_se_out_layer_call_and_return_conditional_losses_42908?
1stack_0_block0_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_0_block0_se_out/PartitionedCall:output:0stack_0_block0_mb_pw_conv_44665*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_MB_pw_conv_layer_call_and_return_conditional_losses_42917?
/stack_0_block0_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_0_block0_MB_pw_conv/StatefulPartitionedCall:output:0stack_0_block0_mb_pw_bn_44668stack_0_block0_mb_pw_bn_44670stack_0_block0_mb_pw_bn_44672stack_0_block0_mb_pw_bn_44674*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42938?
%stack_0_block0_output/PartitionedCallPartitionedCall8stack_0_block0_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_output_layer_call_and_return_conditional_losses_42952?
-stack_1_block0_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall.stack_0_block0_output/PartitionedCall:output:0stack_1_block0_mb_dw__44678*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_MB_dw__layer_call_and_return_conditional_losses_42963?
/stack_1_block0_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_1_block0_MB_dw_/StatefulPartitionedCall:output:0stack_1_block0_mb_dw_bn_44681stack_1_block0_mb_dw_bn_44683stack_1_block0_mb_dw_bn_44685stack_1_block0_mb_dw_bn_44687*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42984?
*stack_1_block0_MB_dw_swish/PartitionedCallPartitionedCall8stack_1_block0_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_43004}
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_1/MeanMean3stack_1_block0_MB_dw_swish/PartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
0stack_1_block0_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_1/Mean:output:0stack_1_block0_se_1_conv_44693stack_1_block0_se_1_conv_44695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_1_conv_layer_call_and_return_conditional_losses_43018?
'stack_1_block0_se_swish/PartitionedCallPartitionedCall9stack_1_block0_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_43034?
0stack_1_block0_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_1_block0_se_swish/PartitionedCall:output:0stack_1_block0_se_2_conv_44699stack_1_block0_se_2_conv_44701*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_2_conv_layer_call_and_return_conditional_losses_43046?
)stack_1_block0_se_sigmoid/PartitionedCallPartitionedCall9stack_1_block0_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_se_sigmoid_layer_call_and_return_conditional_losses_43057?
%stack_1_block0_se_out/PartitionedCallPartitionedCall3stack_1_block0_MB_dw_swish/PartitionedCall:output:02stack_1_block0_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_se_out_layer_call_and_return_conditional_losses_43065?
1stack_1_block0_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block0_se_out/PartitionedCall:output:0stack_1_block0_mb_pw_conv_44706*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_MB_pw_conv_layer_call_and_return_conditional_losses_43074?
/stack_1_block0_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_1_block0_MB_pw_conv/StatefulPartitionedCall:output:0stack_1_block0_mb_pw_bn_44709stack_1_block0_mb_pw_bn_44711stack_1_block0_mb_pw_bn_44713stack_1_block0_mb_pw_bn_44715*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43095?
%stack_1_block0_output/PartitionedCallPartitionedCall8stack_1_block0_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_output_layer_call_and_return_conditional_losses_43109?
-stack_1_block1_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block0_output/PartitionedCall:output:0stack_1_block1_mb_dw__44719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_MB_dw__layer_call_and_return_conditional_losses_43120?
/stack_1_block1_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_1_block1_MB_dw_/StatefulPartitionedCall:output:0stack_1_block1_mb_dw_bn_44722stack_1_block1_mb_dw_bn_44724stack_1_block1_mb_dw_bn_44726stack_1_block1_mb_dw_bn_44728*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_43141?
*stack_1_block1_MB_dw_swish/PartitionedCallPartitionedCall8stack_1_block1_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_43161}
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_2/MeanMean3stack_1_block1_MB_dw_swish/PartitionedCall:output:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
0stack_1_block1_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_2/Mean:output:0stack_1_block1_se_1_conv_44734stack_1_block1_se_1_conv_44736*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_1_conv_layer_call_and_return_conditional_losses_43175?
'stack_1_block1_se_swish/PartitionedCallPartitionedCall9stack_1_block1_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_43191?
0stack_1_block1_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_1_block1_se_swish/PartitionedCall:output:0stack_1_block1_se_2_conv_44740stack_1_block1_se_2_conv_44742*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_2_conv_layer_call_and_return_conditional_losses_43203?
)stack_1_block1_se_sigmoid/PartitionedCallPartitionedCall9stack_1_block1_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_se_sigmoid_layer_call_and_return_conditional_losses_43214?
%stack_1_block1_se_out/PartitionedCallPartitionedCall3stack_1_block1_MB_dw_swish/PartitionedCall:output:02stack_1_block1_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_se_out_layer_call_and_return_conditional_losses_43222?
1stack_1_block1_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block1_se_out/PartitionedCall:output:0stack_1_block1_mb_pw_conv_44747*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_MB_pw_conv_layer_call_and_return_conditional_losses_43231?
/stack_1_block1_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_1_block1_MB_pw_conv/StatefulPartitionedCall:output:0stack_1_block1_mb_pw_bn_44750stack_1_block1_mb_pw_bn_44752stack_1_block1_mb_pw_bn_44754stack_1_block1_mb_pw_bn_44756*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_43252?
'stack_1_block1_dropdrop/PartitionedCallPartitionedCall8stack_1_block1_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_43267?
%stack_1_block1_output/PartitionedCallPartitionedCall.stack_1_block0_output/PartitionedCall:output:00stack_1_block1_dropdrop/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_output_layer_call_and_return_conditional_losses_43275?
!post_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block1_output/PartitionedCall:output:0post_conv_44761*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_post_conv_layer_call_and_return_conditional_losses_43284?
post_bn/StatefulPartitionedCallStatefulPartitionedCall*post_conv/StatefulPartitionedCall:output:0post_bn_44764post_bn_44766post_bn_44768post_bn_44770*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_post_bn_layer_call_and_return_conditional_losses_43305?
post_swish/PartitionedCallPartitionedCall(post_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_post_swish_layer_call_and_return_conditional_losses_43325?
avg_pool/PartitionedCallPartitionedCall#post_swish/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_43332?
head_drop/PartitionedCallPartitionedCall!avg_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_head_drop_layer_call_and_return_conditional_losses_43339?
#predictions/StatefulPartitionedCallStatefulPartitionedCall"head_drop/PartitionedCall:output:0predictions_44776predictions_44778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_43352{
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp ^post_bn/StatefulPartitionedCall"^post_conv/StatefulPartitionedCall$^predictions/StatefulPartitionedCall.^stack_0_block0_MB_dw_/StatefulPartitionedCall0^stack_0_block0_MB_dw_bn/StatefulPartitionedCall0^stack_0_block0_MB_pw_bn/StatefulPartitionedCall2^stack_0_block0_MB_pw_conv/StatefulPartitionedCall1^stack_0_block0_se_1_conv/StatefulPartitionedCall1^stack_0_block0_se_2_conv/StatefulPartitionedCall.^stack_1_block0_MB_dw_/StatefulPartitionedCall0^stack_1_block0_MB_dw_bn/StatefulPartitionedCall0^stack_1_block0_MB_pw_bn/StatefulPartitionedCall2^stack_1_block0_MB_pw_conv/StatefulPartitionedCall1^stack_1_block0_se_1_conv/StatefulPartitionedCall1^stack_1_block0_se_2_conv/StatefulPartitionedCall.^stack_1_block1_MB_dw_/StatefulPartitionedCall0^stack_1_block1_MB_dw_bn/StatefulPartitionedCall0^stack_1_block1_MB_pw_bn/StatefulPartitionedCall2^stack_1_block1_MB_pw_conv/StatefulPartitionedCall1^stack_1_block1_se_1_conv/StatefulPartitionedCall1^stack_1_block1_se_2_conv/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
post_bn/StatefulPartitionedCallpost_bn/StatefulPartitionedCall2F
!post_conv/StatefulPartitionedCall!post_conv/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall2^
-stack_0_block0_MB_dw_/StatefulPartitionedCall-stack_0_block0_MB_dw_/StatefulPartitionedCall2b
/stack_0_block0_MB_dw_bn/StatefulPartitionedCall/stack_0_block0_MB_dw_bn/StatefulPartitionedCall2b
/stack_0_block0_MB_pw_bn/StatefulPartitionedCall/stack_0_block0_MB_pw_bn/StatefulPartitionedCall2f
1stack_0_block0_MB_pw_conv/StatefulPartitionedCall1stack_0_block0_MB_pw_conv/StatefulPartitionedCall2d
0stack_0_block0_se_1_conv/StatefulPartitionedCall0stack_0_block0_se_1_conv/StatefulPartitionedCall2d
0stack_0_block0_se_2_conv/StatefulPartitionedCall0stack_0_block0_se_2_conv/StatefulPartitionedCall2^
-stack_1_block0_MB_dw_/StatefulPartitionedCall-stack_1_block0_MB_dw_/StatefulPartitionedCall2b
/stack_1_block0_MB_dw_bn/StatefulPartitionedCall/stack_1_block0_MB_dw_bn/StatefulPartitionedCall2b
/stack_1_block0_MB_pw_bn/StatefulPartitionedCall/stack_1_block0_MB_pw_bn/StatefulPartitionedCall2f
1stack_1_block0_MB_pw_conv/StatefulPartitionedCall1stack_1_block0_MB_pw_conv/StatefulPartitionedCall2d
0stack_1_block0_se_1_conv/StatefulPartitionedCall0stack_1_block0_se_1_conv/StatefulPartitionedCall2d
0stack_1_block0_se_2_conv/StatefulPartitionedCall0stack_1_block0_se_2_conv/StatefulPartitionedCall2^
-stack_1_block1_MB_dw_/StatefulPartitionedCall-stack_1_block1_MB_dw_/StatefulPartitionedCall2b
/stack_1_block1_MB_dw_bn/StatefulPartitionedCall/stack_1_block1_MB_dw_bn/StatefulPartitionedCall2b
/stack_1_block1_MB_pw_bn/StatefulPartitionedCall/stack_1_block1_MB_pw_bn/StatefulPartitionedCall2f
1stack_1_block1_MB_pw_conv/StatefulPartitionedCall1stack_1_block1_MB_pw_conv/StatefulPartitionedCall2d
0stack_1_block1_se_1_conv/StatefulPartitionedCall0stack_1_block1_se_1_conv/StatefulPartitionedCall2d
0stack_1_block1_se_2_conv/StatefulPartitionedCall0stack_1_block1_se_2_conv/StatefulPartitionedCall2B
stem_bn/StatefulPartitionedCallstem_bn/StatefulPartitionedCall2F
!stem_conv/StatefulPartitionedCall!stem_conv/StatefulPartitionedCall:Y U
0
_output_shapes
:????????? ?
!
_user_specified_name	input_1
?
?
T__inference_stack_1_block1_MB_pw_conv_layer_call_and_return_conditional_losses_46971

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????`^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
z
P__inference_stack_1_block1_output_layer_call_and_return_conditional_losses_43275

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:?????????`W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????`:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
,__inference_EfficientNet_layer_call_fn_45177

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: #
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: 

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:$

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:$

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:$

unknown_37:

unknown_38:$

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:%

unknown_46:?


unknown_47:	?


unknown_48:	?


unknown_49:	?


unknown_50:	?


unknown_51:	?


unknown_52:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_EfficientNet_layer_call_and_return_conditional_losses_43359o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
z
P__inference_stack_1_block0_se_out_layer_call_and_return_conditional_losses_43065

inputs
inputs_1
identityV
mulMulinputsinputs_1*
T0*/
_output_shapes
:?????????`W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????`:?????????:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46418

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_43889w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
p
R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_46916

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:??????????
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-46909*J
_output_shapes8
6:?????????:?????????d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46682

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46454

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_post_bn_layer_call_and_return_conditional_losses_47262

inputs&
readvariableop_resource:	?
(
readvariableop_1_resource:	?
7
(fusedbatchnormv3_readvariableop_resource:	?
9
*fusedbatchnormv3_readvariableop_1_resource:	?

identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?
*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????`?
:?
:?
:?
:?
:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????`?
?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????`?
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?
?
B__inference_stem_bn_layer_call_and_return_conditional_losses_45922

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46095

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_43889

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
s
U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_46505

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????`Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????`W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????`?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-46498*J
_output_shapes8
6:?????????`:?????????`d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47041

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
S__inference_stack_1_block1_se_2_conv_layer_call_and_return_conditional_losses_43203

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43947

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
S
7__inference_stack_0_block0_se_swish_layer_call_fn_46152

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_42877h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
5__inference_stack_1_block0_MB_dw__layer_call_fn_46357

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_MB_dw__layer_call_and_return_conditional_losses_42963w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
c
D__inference_head_drop_layer_call_and_return_conditional_losses_47344

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????
p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????
j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????
Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
_
C__inference_avg_pool_layer_call_and_return_conditional_losses_42737

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47661
result_grads_0
result_grads_11
-sigmoid_efficientnet_stem_bn_fusedbatchnormv3
identity?
SigmoidSigmoid-sigmoid_efficientnet_stem_bn_fusedbatchnormv3^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? }
mulMul-sigmoid_efficientnet_stem_bn_fusedbatchnormv3sub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
?
z
P__inference_stack_1_block1_se_out_layer_call_and_return_conditional_losses_43222

inputs
inputs_1
identityV
mulMulinputsinputs_1*
T0*/
_output_shapes
:?????????`W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????`:?????????:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
P__inference_stack_0_block0_output_layer_call_and_return_conditional_losses_42952

inputs
identityW
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_42652

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
S__inference_stack_1_block1_se_2_conv_layer_call_and_return_conditional_losses_46935

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_avg_pool_layer_call_and_return_conditional_losses_43332

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      h
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????
V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????`?
:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_48051
result_grads_0
result_grads_1,
(sigmoid_stack_0_block0_se_1_conv_biasadd
identity?
SigmoidSigmoid(sigmoid_stack_0_block0_se_1_conv_biasadd^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????w
mulMul(sigmoid_stack_0_block0_se_1_conv_biasaddsub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
U
9__inference_stack_0_block0_se_sigmoid_layer_call_fn_46186

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_se_sigmoid_layer_call_and_return_conditional_losses_42900h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?E
__inference__traced_save_48404
file_prefix/
+savev2_stem_conv_kernel_read_readvariableop,
(savev2_stem_bn_gamma_read_readvariableop+
'savev2_stem_bn_beta_read_readvariableop2
.savev2_stem_bn_moving_mean_read_readvariableop6
2savev2_stem_bn_moving_variance_read_readvariableopE
Asavev2_stack_0_block0_mb_dw__depthwise_kernel_read_readvariableop<
8savev2_stack_0_block0_mb_dw_bn_gamma_read_readvariableop;
7savev2_stack_0_block0_mb_dw_bn_beta_read_readvariableopB
>savev2_stack_0_block0_mb_dw_bn_moving_mean_read_readvariableopF
Bsavev2_stack_0_block0_mb_dw_bn_moving_variance_read_readvariableop>
:savev2_stack_0_block0_se_1_conv_kernel_read_readvariableop<
8savev2_stack_0_block0_se_1_conv_bias_read_readvariableop>
:savev2_stack_0_block0_se_2_conv_kernel_read_readvariableop<
8savev2_stack_0_block0_se_2_conv_bias_read_readvariableop?
;savev2_stack_0_block0_mb_pw_conv_kernel_read_readvariableop<
8savev2_stack_0_block0_mb_pw_bn_gamma_read_readvariableop;
7savev2_stack_0_block0_mb_pw_bn_beta_read_readvariableopB
>savev2_stack_0_block0_mb_pw_bn_moving_mean_read_readvariableopF
Bsavev2_stack_0_block0_mb_pw_bn_moving_variance_read_readvariableopE
Asavev2_stack_1_block0_mb_dw__depthwise_kernel_read_readvariableop<
8savev2_stack_1_block0_mb_dw_bn_gamma_read_readvariableop;
7savev2_stack_1_block0_mb_dw_bn_beta_read_readvariableopB
>savev2_stack_1_block0_mb_dw_bn_moving_mean_read_readvariableopF
Bsavev2_stack_1_block0_mb_dw_bn_moving_variance_read_readvariableop>
:savev2_stack_1_block0_se_1_conv_kernel_read_readvariableop<
8savev2_stack_1_block0_se_1_conv_bias_read_readvariableop>
:savev2_stack_1_block0_se_2_conv_kernel_read_readvariableop<
8savev2_stack_1_block0_se_2_conv_bias_read_readvariableop?
;savev2_stack_1_block0_mb_pw_conv_kernel_read_readvariableop<
8savev2_stack_1_block0_mb_pw_bn_gamma_read_readvariableop;
7savev2_stack_1_block0_mb_pw_bn_beta_read_readvariableopB
>savev2_stack_1_block0_mb_pw_bn_moving_mean_read_readvariableopF
Bsavev2_stack_1_block0_mb_pw_bn_moving_variance_read_readvariableopE
Asavev2_stack_1_block1_mb_dw__depthwise_kernel_read_readvariableop<
8savev2_stack_1_block1_mb_dw_bn_gamma_read_readvariableop;
7savev2_stack_1_block1_mb_dw_bn_beta_read_readvariableopB
>savev2_stack_1_block1_mb_dw_bn_moving_mean_read_readvariableopF
Bsavev2_stack_1_block1_mb_dw_bn_moving_variance_read_readvariableop>
:savev2_stack_1_block1_se_1_conv_kernel_read_readvariableop<
8savev2_stack_1_block1_se_1_conv_bias_read_readvariableop>
:savev2_stack_1_block1_se_2_conv_kernel_read_readvariableop<
8savev2_stack_1_block1_se_2_conv_bias_read_readvariableop?
;savev2_stack_1_block1_mb_pw_conv_kernel_read_readvariableop<
8savev2_stack_1_block1_mb_pw_bn_gamma_read_readvariableop;
7savev2_stack_1_block1_mb_pw_bn_beta_read_readvariableopB
>savev2_stack_1_block1_mb_pw_bn_moving_mean_read_readvariableopF
Bsavev2_stack_1_block1_mb_pw_bn_moving_variance_read_readvariableop/
+savev2_post_conv_kernel_read_readvariableop,
(savev2_post_bn_gamma_read_readvariableop+
'savev2_post_bn_beta_read_readvariableop2
.savev2_post_bn_moving_mean_read_readvariableop6
2savev2_post_bn_moving_variance_read_readvariableop1
-savev2_predictions_kernel_read_readvariableop/
+savev2_predictions_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_stem_conv_kernel_m_read_readvariableop3
/savev2_adam_stem_bn_gamma_m_read_readvariableop2
.savev2_adam_stem_bn_beta_m_read_readvariableopL
Hsavev2_adam_stack_0_block0_mb_dw__depthwise_kernel_m_read_readvariableopC
?savev2_adam_stack_0_block0_mb_dw_bn_gamma_m_read_readvariableopB
>savev2_adam_stack_0_block0_mb_dw_bn_beta_m_read_readvariableopE
Asavev2_adam_stack_0_block0_se_1_conv_kernel_m_read_readvariableopC
?savev2_adam_stack_0_block0_se_1_conv_bias_m_read_readvariableopE
Asavev2_adam_stack_0_block0_se_2_conv_kernel_m_read_readvariableopC
?savev2_adam_stack_0_block0_se_2_conv_bias_m_read_readvariableopF
Bsavev2_adam_stack_0_block0_mb_pw_conv_kernel_m_read_readvariableopC
?savev2_adam_stack_0_block0_mb_pw_bn_gamma_m_read_readvariableopB
>savev2_adam_stack_0_block0_mb_pw_bn_beta_m_read_readvariableopL
Hsavev2_adam_stack_1_block0_mb_dw__depthwise_kernel_m_read_readvariableopC
?savev2_adam_stack_1_block0_mb_dw_bn_gamma_m_read_readvariableopB
>savev2_adam_stack_1_block0_mb_dw_bn_beta_m_read_readvariableopE
Asavev2_adam_stack_1_block0_se_1_conv_kernel_m_read_readvariableopC
?savev2_adam_stack_1_block0_se_1_conv_bias_m_read_readvariableopE
Asavev2_adam_stack_1_block0_se_2_conv_kernel_m_read_readvariableopC
?savev2_adam_stack_1_block0_se_2_conv_bias_m_read_readvariableopF
Bsavev2_adam_stack_1_block0_mb_pw_conv_kernel_m_read_readvariableopC
?savev2_adam_stack_1_block0_mb_pw_bn_gamma_m_read_readvariableopB
>savev2_adam_stack_1_block0_mb_pw_bn_beta_m_read_readvariableopL
Hsavev2_adam_stack_1_block1_mb_dw__depthwise_kernel_m_read_readvariableopC
?savev2_adam_stack_1_block1_mb_dw_bn_gamma_m_read_readvariableopB
>savev2_adam_stack_1_block1_mb_dw_bn_beta_m_read_readvariableopE
Asavev2_adam_stack_1_block1_se_1_conv_kernel_m_read_readvariableopC
?savev2_adam_stack_1_block1_se_1_conv_bias_m_read_readvariableopE
Asavev2_adam_stack_1_block1_se_2_conv_kernel_m_read_readvariableopC
?savev2_adam_stack_1_block1_se_2_conv_bias_m_read_readvariableopF
Bsavev2_adam_stack_1_block1_mb_pw_conv_kernel_m_read_readvariableopC
?savev2_adam_stack_1_block1_mb_pw_bn_gamma_m_read_readvariableopB
>savev2_adam_stack_1_block1_mb_pw_bn_beta_m_read_readvariableop6
2savev2_adam_post_conv_kernel_m_read_readvariableop3
/savev2_adam_post_bn_gamma_m_read_readvariableop2
.savev2_adam_post_bn_beta_m_read_readvariableop8
4savev2_adam_predictions_kernel_m_read_readvariableop6
2savev2_adam_predictions_bias_m_read_readvariableop6
2savev2_adam_stem_conv_kernel_v_read_readvariableop3
/savev2_adam_stem_bn_gamma_v_read_readvariableop2
.savev2_adam_stem_bn_beta_v_read_readvariableopL
Hsavev2_adam_stack_0_block0_mb_dw__depthwise_kernel_v_read_readvariableopC
?savev2_adam_stack_0_block0_mb_dw_bn_gamma_v_read_readvariableopB
>savev2_adam_stack_0_block0_mb_dw_bn_beta_v_read_readvariableopE
Asavev2_adam_stack_0_block0_se_1_conv_kernel_v_read_readvariableopC
?savev2_adam_stack_0_block0_se_1_conv_bias_v_read_readvariableopE
Asavev2_adam_stack_0_block0_se_2_conv_kernel_v_read_readvariableopC
?savev2_adam_stack_0_block0_se_2_conv_bias_v_read_readvariableopF
Bsavev2_adam_stack_0_block0_mb_pw_conv_kernel_v_read_readvariableopC
?savev2_adam_stack_0_block0_mb_pw_bn_gamma_v_read_readvariableopB
>savev2_adam_stack_0_block0_mb_pw_bn_beta_v_read_readvariableopL
Hsavev2_adam_stack_1_block0_mb_dw__depthwise_kernel_v_read_readvariableopC
?savev2_adam_stack_1_block0_mb_dw_bn_gamma_v_read_readvariableopB
>savev2_adam_stack_1_block0_mb_dw_bn_beta_v_read_readvariableopE
Asavev2_adam_stack_1_block0_se_1_conv_kernel_v_read_readvariableopC
?savev2_adam_stack_1_block0_se_1_conv_bias_v_read_readvariableopE
Asavev2_adam_stack_1_block0_se_2_conv_kernel_v_read_readvariableopC
?savev2_adam_stack_1_block0_se_2_conv_bias_v_read_readvariableopF
Bsavev2_adam_stack_1_block0_mb_pw_conv_kernel_v_read_readvariableopC
?savev2_adam_stack_1_block0_mb_pw_bn_gamma_v_read_readvariableopB
>savev2_adam_stack_1_block0_mb_pw_bn_beta_v_read_readvariableopL
Hsavev2_adam_stack_1_block1_mb_dw__depthwise_kernel_v_read_readvariableopC
?savev2_adam_stack_1_block1_mb_dw_bn_gamma_v_read_readvariableopB
>savev2_adam_stack_1_block1_mb_dw_bn_beta_v_read_readvariableopE
Asavev2_adam_stack_1_block1_se_1_conv_kernel_v_read_readvariableopC
?savev2_adam_stack_1_block1_se_1_conv_bias_v_read_readvariableopE
Asavev2_adam_stack_1_block1_se_2_conv_kernel_v_read_readvariableopC
?savev2_adam_stack_1_block1_se_2_conv_bias_v_read_readvariableopF
Bsavev2_adam_stack_1_block1_mb_pw_conv_kernel_v_read_readvariableopC
?savev2_adam_stack_1_block1_mb_pw_bn_gamma_v_read_readvariableopB
>savev2_adam_stack_1_block1_mb_pw_bn_beta_v_read_readvariableop6
2savev2_adam_post_conv_kernel_v_read_readvariableop3
/savev2_adam_post_bn_gamma_v_read_readvariableop2
.savev2_adam_post_bn_beta_v_read_readvariableop8
4savev2_adam_predictions_kernel_v_read_readvariableop6
2savev2_adam_predictions_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?O
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?N
value?NB?N?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-8/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-19/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-19/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-19/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-19/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-8/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB]layer_with_weights-14/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-19/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?B
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_stem_conv_kernel_read_readvariableop(savev2_stem_bn_gamma_read_readvariableop'savev2_stem_bn_beta_read_readvariableop.savev2_stem_bn_moving_mean_read_readvariableop2savev2_stem_bn_moving_variance_read_readvariableopAsavev2_stack_0_block0_mb_dw__depthwise_kernel_read_readvariableop8savev2_stack_0_block0_mb_dw_bn_gamma_read_readvariableop7savev2_stack_0_block0_mb_dw_bn_beta_read_readvariableop>savev2_stack_0_block0_mb_dw_bn_moving_mean_read_readvariableopBsavev2_stack_0_block0_mb_dw_bn_moving_variance_read_readvariableop:savev2_stack_0_block0_se_1_conv_kernel_read_readvariableop8savev2_stack_0_block0_se_1_conv_bias_read_readvariableop:savev2_stack_0_block0_se_2_conv_kernel_read_readvariableop8savev2_stack_0_block0_se_2_conv_bias_read_readvariableop;savev2_stack_0_block0_mb_pw_conv_kernel_read_readvariableop8savev2_stack_0_block0_mb_pw_bn_gamma_read_readvariableop7savev2_stack_0_block0_mb_pw_bn_beta_read_readvariableop>savev2_stack_0_block0_mb_pw_bn_moving_mean_read_readvariableopBsavev2_stack_0_block0_mb_pw_bn_moving_variance_read_readvariableopAsavev2_stack_1_block0_mb_dw__depthwise_kernel_read_readvariableop8savev2_stack_1_block0_mb_dw_bn_gamma_read_readvariableop7savev2_stack_1_block0_mb_dw_bn_beta_read_readvariableop>savev2_stack_1_block0_mb_dw_bn_moving_mean_read_readvariableopBsavev2_stack_1_block0_mb_dw_bn_moving_variance_read_readvariableop:savev2_stack_1_block0_se_1_conv_kernel_read_readvariableop8savev2_stack_1_block0_se_1_conv_bias_read_readvariableop:savev2_stack_1_block0_se_2_conv_kernel_read_readvariableop8savev2_stack_1_block0_se_2_conv_bias_read_readvariableop;savev2_stack_1_block0_mb_pw_conv_kernel_read_readvariableop8savev2_stack_1_block0_mb_pw_bn_gamma_read_readvariableop7savev2_stack_1_block0_mb_pw_bn_beta_read_readvariableop>savev2_stack_1_block0_mb_pw_bn_moving_mean_read_readvariableopBsavev2_stack_1_block0_mb_pw_bn_moving_variance_read_readvariableopAsavev2_stack_1_block1_mb_dw__depthwise_kernel_read_readvariableop8savev2_stack_1_block1_mb_dw_bn_gamma_read_readvariableop7savev2_stack_1_block1_mb_dw_bn_beta_read_readvariableop>savev2_stack_1_block1_mb_dw_bn_moving_mean_read_readvariableopBsavev2_stack_1_block1_mb_dw_bn_moving_variance_read_readvariableop:savev2_stack_1_block1_se_1_conv_kernel_read_readvariableop8savev2_stack_1_block1_se_1_conv_bias_read_readvariableop:savev2_stack_1_block1_se_2_conv_kernel_read_readvariableop8savev2_stack_1_block1_se_2_conv_bias_read_readvariableop;savev2_stack_1_block1_mb_pw_conv_kernel_read_readvariableop8savev2_stack_1_block1_mb_pw_bn_gamma_read_readvariableop7savev2_stack_1_block1_mb_pw_bn_beta_read_readvariableop>savev2_stack_1_block1_mb_pw_bn_moving_mean_read_readvariableopBsavev2_stack_1_block1_mb_pw_bn_moving_variance_read_readvariableop+savev2_post_conv_kernel_read_readvariableop(savev2_post_bn_gamma_read_readvariableop'savev2_post_bn_beta_read_readvariableop.savev2_post_bn_moving_mean_read_readvariableop2savev2_post_bn_moving_variance_read_readvariableop-savev2_predictions_kernel_read_readvariableop+savev2_predictions_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_stem_conv_kernel_m_read_readvariableop/savev2_adam_stem_bn_gamma_m_read_readvariableop.savev2_adam_stem_bn_beta_m_read_readvariableopHsavev2_adam_stack_0_block0_mb_dw__depthwise_kernel_m_read_readvariableop?savev2_adam_stack_0_block0_mb_dw_bn_gamma_m_read_readvariableop>savev2_adam_stack_0_block0_mb_dw_bn_beta_m_read_readvariableopAsavev2_adam_stack_0_block0_se_1_conv_kernel_m_read_readvariableop?savev2_adam_stack_0_block0_se_1_conv_bias_m_read_readvariableopAsavev2_adam_stack_0_block0_se_2_conv_kernel_m_read_readvariableop?savev2_adam_stack_0_block0_se_2_conv_bias_m_read_readvariableopBsavev2_adam_stack_0_block0_mb_pw_conv_kernel_m_read_readvariableop?savev2_adam_stack_0_block0_mb_pw_bn_gamma_m_read_readvariableop>savev2_adam_stack_0_block0_mb_pw_bn_beta_m_read_readvariableopHsavev2_adam_stack_1_block0_mb_dw__depthwise_kernel_m_read_readvariableop?savev2_adam_stack_1_block0_mb_dw_bn_gamma_m_read_readvariableop>savev2_adam_stack_1_block0_mb_dw_bn_beta_m_read_readvariableopAsavev2_adam_stack_1_block0_se_1_conv_kernel_m_read_readvariableop?savev2_adam_stack_1_block0_se_1_conv_bias_m_read_readvariableopAsavev2_adam_stack_1_block0_se_2_conv_kernel_m_read_readvariableop?savev2_adam_stack_1_block0_se_2_conv_bias_m_read_readvariableopBsavev2_adam_stack_1_block0_mb_pw_conv_kernel_m_read_readvariableop?savev2_adam_stack_1_block0_mb_pw_bn_gamma_m_read_readvariableop>savev2_adam_stack_1_block0_mb_pw_bn_beta_m_read_readvariableopHsavev2_adam_stack_1_block1_mb_dw__depthwise_kernel_m_read_readvariableop?savev2_adam_stack_1_block1_mb_dw_bn_gamma_m_read_readvariableop>savev2_adam_stack_1_block1_mb_dw_bn_beta_m_read_readvariableopAsavev2_adam_stack_1_block1_se_1_conv_kernel_m_read_readvariableop?savev2_adam_stack_1_block1_se_1_conv_bias_m_read_readvariableopAsavev2_adam_stack_1_block1_se_2_conv_kernel_m_read_readvariableop?savev2_adam_stack_1_block1_se_2_conv_bias_m_read_readvariableopBsavev2_adam_stack_1_block1_mb_pw_conv_kernel_m_read_readvariableop?savev2_adam_stack_1_block1_mb_pw_bn_gamma_m_read_readvariableop>savev2_adam_stack_1_block1_mb_pw_bn_beta_m_read_readvariableop2savev2_adam_post_conv_kernel_m_read_readvariableop/savev2_adam_post_bn_gamma_m_read_readvariableop.savev2_adam_post_bn_beta_m_read_readvariableop4savev2_adam_predictions_kernel_m_read_readvariableop2savev2_adam_predictions_bias_m_read_readvariableop2savev2_adam_stem_conv_kernel_v_read_readvariableop/savev2_adam_stem_bn_gamma_v_read_readvariableop.savev2_adam_stem_bn_beta_v_read_readvariableopHsavev2_adam_stack_0_block0_mb_dw__depthwise_kernel_v_read_readvariableop?savev2_adam_stack_0_block0_mb_dw_bn_gamma_v_read_readvariableop>savev2_adam_stack_0_block0_mb_dw_bn_beta_v_read_readvariableopAsavev2_adam_stack_0_block0_se_1_conv_kernel_v_read_readvariableop?savev2_adam_stack_0_block0_se_1_conv_bias_v_read_readvariableopAsavev2_adam_stack_0_block0_se_2_conv_kernel_v_read_readvariableop?savev2_adam_stack_0_block0_se_2_conv_bias_v_read_readvariableopBsavev2_adam_stack_0_block0_mb_pw_conv_kernel_v_read_readvariableop?savev2_adam_stack_0_block0_mb_pw_bn_gamma_v_read_readvariableop>savev2_adam_stack_0_block0_mb_pw_bn_beta_v_read_readvariableopHsavev2_adam_stack_1_block0_mb_dw__depthwise_kernel_v_read_readvariableop?savev2_adam_stack_1_block0_mb_dw_bn_gamma_v_read_readvariableop>savev2_adam_stack_1_block0_mb_dw_bn_beta_v_read_readvariableopAsavev2_adam_stack_1_block0_se_1_conv_kernel_v_read_readvariableop?savev2_adam_stack_1_block0_se_1_conv_bias_v_read_readvariableopAsavev2_adam_stack_1_block0_se_2_conv_kernel_v_read_readvariableop?savev2_adam_stack_1_block0_se_2_conv_bias_v_read_readvariableopBsavev2_adam_stack_1_block0_mb_pw_conv_kernel_v_read_readvariableop?savev2_adam_stack_1_block0_mb_pw_bn_gamma_v_read_readvariableop>savev2_adam_stack_1_block0_mb_pw_bn_beta_v_read_readvariableopHsavev2_adam_stack_1_block1_mb_dw__depthwise_kernel_v_read_readvariableop?savev2_adam_stack_1_block1_mb_dw_bn_gamma_v_read_readvariableop>savev2_adam_stack_1_block1_mb_dw_bn_beta_v_read_readvariableopAsavev2_adam_stack_1_block1_se_1_conv_kernel_v_read_readvariableop?savev2_adam_stack_1_block1_se_1_conv_bias_v_read_readvariableopAsavev2_adam_stack_1_block1_se_2_conv_kernel_v_read_readvariableop?savev2_adam_stack_1_block1_se_2_conv_bias_v_read_readvariableopBsavev2_adam_stack_1_block1_mb_pw_conv_kernel_v_read_readvariableop?savev2_adam_stack_1_block1_mb_pw_bn_gamma_v_read_readvariableop>savev2_adam_stack_1_block1_mb_pw_bn_beta_v_read_readvariableop2savev2_adam_post_conv_kernel_v_read_readvariableop/savev2_adam_post_bn_gamma_v_read_readvariableop.savev2_adam_post_bn_beta_v_read_readvariableop4savev2_adam_predictions_kernel_v_read_readvariableop2savev2_adam_predictions_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?

_input_shapes?

?
: : : : : : : : : : : : :: : : :::::::::::::::::::::::::::::::::?
:?
:?
:?
:?
:	?
:: : : : : : : : : : : : : : : : :: : : :::::::::::::::::::::::?
:?
:?
:	?
:: : : : : : : :: : : :::::::::::::::::::::::?
:?
:?
:	?
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::  

_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
::,'(
&
_output_shapes
:: (

_output_shapes
::,)(
&
_output_shapes
:: *

_output_shapes
::,+(
&
_output_shapes
:: ,

_output_shapes
:: -

_output_shapes
:: .

_output_shapes
:: /

_output_shapes
::-0)
'
_output_shapes
:?
:!1

_output_shapes	
:?
:!2

_output_shapes	
:?
:!3

_output_shapes	
:?
:!4

_output_shapes	
:?
:%5!

_output_shapes
:	?
: 6

_output_shapes
::7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :,@(
&
_output_shapes
: : A

_output_shapes
: : B

_output_shapes
: :,C(
&
_output_shapes
: : D

_output_shapes
: : E

_output_shapes
: :,F(
&
_output_shapes
: : G

_output_shapes
::,H(
&
_output_shapes
: : I

_output_shapes
: :,J(
&
_output_shapes
: : K

_output_shapes
:: L

_output_shapes
::,M(
&
_output_shapes
:: N

_output_shapes
:: O

_output_shapes
::,P(
&
_output_shapes
:: Q

_output_shapes
::,R(
&
_output_shapes
:: S

_output_shapes
::,T(
&
_output_shapes
:: U

_output_shapes
:: V

_output_shapes
::,W(
&
_output_shapes
:: X

_output_shapes
:: Y

_output_shapes
::,Z(
&
_output_shapes
:: [

_output_shapes
::,\(
&
_output_shapes
:: ]

_output_shapes
::,^(
&
_output_shapes
:: _

_output_shapes
:: `

_output_shapes
::-a)
'
_output_shapes
:?
:!b

_output_shapes	
:?
:!c

_output_shapes	
:?
:%d!

_output_shapes
:	?
: e

_output_shapes
::,f(
&
_output_shapes
: : g

_output_shapes
: : h

_output_shapes
: :,i(
&
_output_shapes
: : j

_output_shapes
: : k

_output_shapes
: :,l(
&
_output_shapes
: : m

_output_shapes
::,n(
&
_output_shapes
: : o

_output_shapes
: :,p(
&
_output_shapes
: : q

_output_shapes
:: r

_output_shapes
::,s(
&
_output_shapes
:: t

_output_shapes
:: u

_output_shapes
::,v(
&
_output_shapes
:: w

_output_shapes
::,x(
&
_output_shapes
:: y

_output_shapes
::,z(
&
_output_shapes
:: {

_output_shapes
:: |

_output_shapes
::,}(
&
_output_shapes
:: ~

_output_shapes
:: 

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::!?

_output_shapes
::.?)
'
_output_shapes
:?
:"?

_output_shapes	
:?
:"?

_output_shapes	
:?
:&?!

_output_shapes
:	?
:!?

_output_shapes
::?

_output_shapes
: 
?
?
,__inference_EfficientNet_layer_call_fn_44621
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: #
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: 

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:$

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:$

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:$

unknown_37:

unknown_38:$

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:%

unknown_46:?


unknown_47:	?


unknown_48:	?


unknown_49:	?


unknown_50:	?


unknown_51:	?


unknown_52:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&"#$'()*+,-01256*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_EfficientNet_layer_call_and_return_conditional_losses_44397o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:????????? ?
!
_user_specified_name	input_1
?
s
U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_43161

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????`Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????`W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????`?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-43154*J
_output_shapes8
6:?????????`:?????????`d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
B__inference_stem_bn_layer_call_and_return_conditional_losses_45958

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46077

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46633

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43095w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_48081
result_grads_0
result_grads_1,
(sigmoid_stack_1_block0_se_1_conv_biasadd
identity?
SigmoidSigmoid(sigmoid_stack_1_block0_se_1_conv_biasadd^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????w
mulMul(sigmoid_stack_1_block0_se_1_conv_biasaddsub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
?
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_47023

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_43637w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
D__inference_stem_conv_layer_call_and_return_conditional_losses_45834

inputs8
conv2d_readvariableop_resource: 
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????? ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? ?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?	
?
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46379

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42429?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_post_swish_layer_call_and_return_conditional_losses_47295

inputs

identity_1U
SigmoidSigmoidinputs*
T0*0
_output_shapes
:?????????`?
Z
mulMulinputsSigmoid:y:0*
T0*0
_output_shapes
:?????????`?
X
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:?????????`?
?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-47288*L
_output_shapes:
8:?????????`?
:?????????`?
e

Identity_1IdentityIdentityN:output:0*
T0*0
_output_shapes
:?????????`?
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????`?
:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_48156
result_grads_0
result_grads_1
sigmoid_inputs
identityn
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ^
mulMulsigmoid_inputssub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
?
?
'__inference_post_bn_layer_call_fn_47208

inputs
unknown:	?

	unknown_0:	?

	unknown_1:	?

	unknown_2:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_post_bn_layer_call_and_return_conditional_losses_43547x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`?
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????`?
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46849

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
B__inference_post_bn_layer_call_and_return_conditional_losses_43305

inputs&
readvariableop_resource:	?
(
readvariableop_1_resource:	?
7
(fusedbatchnormv3_readvariableop_resource:	?
9
*fusedbatchnormv3_readvariableop_1_resource:	?

identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?
*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????`?
:?
:?
:?
:?
:*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????`?
?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????`?
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?
?
B__inference_stem_bn_layer_call_and_return_conditional_losses_45940

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
B__inference_stem_bn_layer_call_and_return_conditional_losses_42237

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
p
T__inference_stack_0_block0_se_sigmoid_layer_call_and_return_conditional_losses_42900

inputs
identityT
SigmoidSigmoidinputs*
T0*/
_output_shapes
:????????? [
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
S__inference_stack_1_block0_se_2_conv_layer_call_and_return_conditional_losses_46558

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_post_conv_layer_call_fn_47149

inputs"
unknown:?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_post_conv_layer_call_and_return_conditional_losses_43284x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`?
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
'__inference_post_bn_layer_call_fn_47195

inputs
unknown:	?

	unknown_0:	?

	unknown_1:	?

	unknown_2:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_post_bn_layer_call_and_return_conditional_losses_43305x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`?
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????`?
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42365

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46664

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
"__inference_internal_grad_fn_47901
result_grads_0
result_grads_1$
 sigmoid_stem_bn_fusedbatchnormv3
identity?
SigmoidSigmoid sigmoid_stem_bn_fusedbatchnormv3^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? p
mulMul sigmoid_stem_bn_fusedbatchnormv3sub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
?
?
"__inference_internal_grad_fn_48096
result_grads_0
result_grads_14
0sigmoid_stack_1_block1_mb_dw_bn_fusedbatchnormv3
identity?
SigmoidSigmoid0sigmoid_stack_1_block1_mb_dw_bn_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`
mulMul0sigmoid_stack_1_block1_mb_dw_bn_fusedbatchnormv3sub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?
?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46472

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46256

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42938x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47077

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47991
result_grads_0
result_grads_1,
(sigmoid_stack_1_block1_se_1_conv_biasadd
identity?
SigmoidSigmoid(sigmoid_stack_1_block1_se_1_conv_biasadd^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????w
mulMul(sigmoid_stack_1_block1_se_1_conv_biasaddsub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?

?
S__inference_stack_1_block0_se_1_conv_layer_call_and_return_conditional_losses_43018

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_post_bn_layer_call_fn_47182

inputs
unknown:	?

	unknown_0:	?

	unknown_1:	?

	unknown_2:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_post_bn_layer_call_and_return_conditional_losses_42716?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????

 
_user_specified_nameinputs
?
V
:__inference_stack_0_block0_MB_dw_swish_layer_call_fn_46118

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_42847i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_47841
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46059

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
+__inference_predictions_layer_call_fn_47353

inputs
unknown:	?

	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_43352o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46795

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_43734w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_48201
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_44044

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
q
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_47130

inputs
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
packed/1Const*
_output_shapes
: *
dtype0*
value	B :J
packed/2Const*
_output_shapes
: *
dtype0*
value	B :J
packed/3Const*
_output_shapes
: *
dtype0*
value	B :?
packedPackstrided_slice:output:0packed/1:output:0packed/2:output:0packed/3:output:0*
N*
T0*
_output_shapes
:R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????`?
$dropout/random_uniform/RandomUniformRandomUniformpacked:output:0*
T0*/
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??*>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????`a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
8__inference_stack_0_block0_se_1_conv_layer_call_fn_46137

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_1_conv_layer_call_and_return_conditional_losses_42861w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
a
5__inference_stack_1_block0_se_out_layer_call_fn_46574
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_se_out_layer_call_and_return_conditional_losses_43065h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????`:?????????:Y U
/
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
p
R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_46162

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:??????????
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-46155*J
_output_shapes8
6:?????????:?????????d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47676
result_grads_0
result_grads_1A
=sigmoid_efficientnet_stack_0_block0_mb_dw_bn_fusedbatchnormv3
identity?
SigmoidSigmoid=sigmoid_efficientnet_stack_0_block0_mb_dw_bn_fusedbatchnormv3^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
mulMul=sigmoid_efficientnet_stack_0_block0_mb_dw_bn_fusedbatchnormv3sub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
?
p
T__inference_stack_1_block1_se_sigmoid_layer_call_and_return_conditional_losses_46945

inputs
identityT
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????[
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_stack_1_block1_MB_pw_conv_layer_call_and_return_conditional_losses_43231

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????`^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
c
E__inference_stem_swish_layer_call_and_return_conditional_losses_45973

inputs

identity_1U
SigmoidSigmoidinputs*
T0*0
_output_shapes
:?????????? Z
mulMulinputsSigmoid:y:0*
T0*0
_output_shapes
:?????????? X
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:?????????? ?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-45966*L
_output_shapes:
8:?????????? :?????????? e

Identity_1IdentityIdentityN:output:0*
T0*0
_output_shapes
:?????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
5__inference_stack_1_block1_MB_dw__layer_call_fn_46734

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_MB_dw__layer_call_and_return_conditional_losses_43120w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

?
"__inference_internal_grad_fn_48126
result_grads_0
result_grads_1$
 sigmoid_post_bn_fusedbatchnormv3
identity?
SigmoidSigmoid sigmoid_post_bn_fusedbatchnormv3^result_grads_0*
T0*0
_output_shapes
:?????????`?
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????`?
p
mulMul sigmoid_post_bn_fusedbatchnormv3sub:z:0*
T0*0
_output_shapes
:?????????`?
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????`?
]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????`?
b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????`?
Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????`?
"
identityIdentity:output:0*g
_input_shapesV
T:?????????`?
:?????????`?
:?????????`?
:` \
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????`?

?

?
F__inference_predictions_layer_call_and_return_conditional_losses_43352

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
s
U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_42847

inputs

identity_1U
SigmoidSigmoidinputs*
T0*0
_output_shapes
:?????????? Z
mulMulinputsSigmoid:y:0*
T0*0
_output_shapes
:?????????? X
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:?????????? ?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-42840*L
_output_shapes:
8:?????????? :?????????? e

Identity_1IdentityIdentityN:output:0*
T0*0
_output_shapes
:?????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
F
*__inference_stem_swish_layer_call_fn_45963

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_stem_swish_layer_call_and_return_conditional_losses_42795i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
'__inference_post_bn_layer_call_fn_47169

inputs
unknown:	?

	unknown_0:	?

	unknown_1:	?

	unknown_2:	?

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_post_bn_layer_call_and_return_conditional_losses_42685?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????
: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????

 
_user_specified_nameinputs
?	
?
P__inference_stack_1_block0_MB_dw__layer_call_and_return_conditional_losses_42963

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
i
IdentityIdentitydepthwise:output:0^NoOp*
T0*/
_output_shapes
:?????????`a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42524

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_stem_bn_layer_call_and_return_conditional_losses_44102

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42984

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?	
?
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_46984

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_42621?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
l
P__inference_stack_1_block0_output_layer_call_and_return_conditional_losses_43109

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
p
R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_42877

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:??????????
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-42870*J
_output_shapes8
6:?????????:?????????d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_stem_bn_layer_call_fn_45847

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_42237?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
a
5__inference_stack_0_block0_se_out_layer_call_fn_46197
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_se_out_layer_call_and_return_conditional_losses_42908i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????? :????????? :Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
b
)__inference_head_drop_layer_call_fn_47327

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_head_drop_layer_call_and_return_conditional_losses_43500p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
l
P__inference_stack_0_block0_output_layer_call_and_return_conditional_losses_46350

inputs
identityW
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
G__inference_EfficientNet_layer_call_and_return_conditional_losses_44943
input_1)
stem_conv_44785: 
stem_bn_44788: 
stem_bn_44790: 
stem_bn_44792: 
stem_bn_44794: 5
stack_0_block0_mb_dw__44798: +
stack_0_block0_mb_dw_bn_44801: +
stack_0_block0_mb_dw_bn_44803: +
stack_0_block0_mb_dw_bn_44805: +
stack_0_block0_mb_dw_bn_44807: 8
stack_0_block0_se_1_conv_44813: ,
stack_0_block0_se_1_conv_44815:8
stack_0_block0_se_2_conv_44819: ,
stack_0_block0_se_2_conv_44821: 9
stack_0_block0_mb_pw_conv_44826: +
stack_0_block0_mb_pw_bn_44829:+
stack_0_block0_mb_pw_bn_44831:+
stack_0_block0_mb_pw_bn_44833:+
stack_0_block0_mb_pw_bn_44835:5
stack_1_block0_mb_dw__44839:+
stack_1_block0_mb_dw_bn_44842:+
stack_1_block0_mb_dw_bn_44844:+
stack_1_block0_mb_dw_bn_44846:+
stack_1_block0_mb_dw_bn_44848:8
stack_1_block0_se_1_conv_44854:,
stack_1_block0_se_1_conv_44856:8
stack_1_block0_se_2_conv_44860:,
stack_1_block0_se_2_conv_44862:9
stack_1_block0_mb_pw_conv_44867:+
stack_1_block0_mb_pw_bn_44870:+
stack_1_block0_mb_pw_bn_44872:+
stack_1_block0_mb_pw_bn_44874:+
stack_1_block0_mb_pw_bn_44876:5
stack_1_block1_mb_dw__44880:+
stack_1_block1_mb_dw_bn_44883:+
stack_1_block1_mb_dw_bn_44885:+
stack_1_block1_mb_dw_bn_44887:+
stack_1_block1_mb_dw_bn_44889:8
stack_1_block1_se_1_conv_44895:,
stack_1_block1_se_1_conv_44897:8
stack_1_block1_se_2_conv_44901:,
stack_1_block1_se_2_conv_44903:9
stack_1_block1_mb_pw_conv_44908:+
stack_1_block1_mb_pw_bn_44911:+
stack_1_block1_mb_pw_bn_44913:+
stack_1_block1_mb_pw_bn_44915:+
stack_1_block1_mb_pw_bn_44917:*
post_conv_44922:?

post_bn_44925:	?

post_bn_44927:	?

post_bn_44929:	?

post_bn_44931:	?
$
predictions_44937:	?

predictions_44939:
identity??!head_drop/StatefulPartitionedCall?post_bn/StatefulPartitionedCall?!post_conv/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?-stack_0_block0_MB_dw_/StatefulPartitionedCall?/stack_0_block0_MB_dw_bn/StatefulPartitionedCall?/stack_0_block0_MB_pw_bn/StatefulPartitionedCall?1stack_0_block0_MB_pw_conv/StatefulPartitionedCall?0stack_0_block0_se_1_conv/StatefulPartitionedCall?0stack_0_block0_se_2_conv/StatefulPartitionedCall?-stack_1_block0_MB_dw_/StatefulPartitionedCall?/stack_1_block0_MB_dw_bn/StatefulPartitionedCall?/stack_1_block0_MB_pw_bn/StatefulPartitionedCall?1stack_1_block0_MB_pw_conv/StatefulPartitionedCall?0stack_1_block0_se_1_conv/StatefulPartitionedCall?0stack_1_block0_se_2_conv/StatefulPartitionedCall?-stack_1_block1_MB_dw_/StatefulPartitionedCall?/stack_1_block1_MB_dw_bn/StatefulPartitionedCall?/stack_1_block1_MB_pw_bn/StatefulPartitionedCall?1stack_1_block1_MB_pw_conv/StatefulPartitionedCall?/stack_1_block1_dropdrop/StatefulPartitionedCall?0stack_1_block1_se_1_conv/StatefulPartitionedCall?0stack_1_block1_se_2_conv/StatefulPartitionedCall?stem_bn/StatefulPartitionedCall?!stem_conv/StatefulPartitionedCall?
!stem_conv/StatefulPartitionedCallStatefulPartitionedCallinput_1stem_conv_44785*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_42754?
stem_bn/StatefulPartitionedCallStatefulPartitionedCall*stem_conv/StatefulPartitionedCall:output:0stem_bn_44788stem_bn_44790stem_bn_44792stem_bn_44794*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_44102?
stem_swish/PartitionedCallPartitionedCall(stem_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_stem_swish_layer_call_and_return_conditional_losses_42795?
-stack_0_block0_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall#stem_swish/PartitionedCall:output:0stack_0_block0_mb_dw__44798*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_MB_dw__layer_call_and_return_conditional_losses_42806?
/stack_0_block0_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_0_block0_MB_dw_/StatefulPartitionedCall:output:0stack_0_block0_mb_dw_bn_44801stack_0_block0_mb_dw_bn_44803stack_0_block0_mb_dw_bn_44805stack_0_block0_mb_dw_bn_44807*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_44044?
*stack_0_block0_MB_dw_swish/PartitionedCallPartitionedCall8stack_0_block0_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_42847{
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean/MeanMean3stack_0_block0_MB_dw_swish/PartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(?
0stack_0_block0_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_mean/Mean:output:0stack_0_block0_se_1_conv_44813stack_0_block0_se_1_conv_44815*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_1_conv_layer_call_and_return_conditional_losses_42861?
'stack_0_block0_se_swish/PartitionedCallPartitionedCall9stack_0_block0_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_42877?
0stack_0_block0_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_0_block0_se_swish/PartitionedCall:output:0stack_0_block0_se_2_conv_44819stack_0_block0_se_2_conv_44821*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_2_conv_layer_call_and_return_conditional_losses_42889?
)stack_0_block0_se_sigmoid/PartitionedCallPartitionedCall9stack_0_block0_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_se_sigmoid_layer_call_and_return_conditional_losses_42900?
%stack_0_block0_se_out/PartitionedCallPartitionedCall3stack_0_block0_MB_dw_swish/PartitionedCall:output:02stack_0_block0_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_se_out_layer_call_and_return_conditional_losses_42908?
1stack_0_block0_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_0_block0_se_out/PartitionedCall:output:0stack_0_block0_mb_pw_conv_44826*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_MB_pw_conv_layer_call_and_return_conditional_losses_42917?
/stack_0_block0_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_0_block0_MB_pw_conv/StatefulPartitionedCall:output:0stack_0_block0_mb_pw_bn_44829stack_0_block0_mb_pw_bn_44831stack_0_block0_mb_pw_bn_44833stack_0_block0_mb_pw_bn_44835*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43947?
%stack_0_block0_output/PartitionedCallPartitionedCall8stack_0_block0_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_output_layer_call_and_return_conditional_losses_42952?
-stack_1_block0_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall.stack_0_block0_output/PartitionedCall:output:0stack_1_block0_mb_dw__44839*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_MB_dw__layer_call_and_return_conditional_losses_42963?
/stack_1_block0_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_1_block0_MB_dw_/StatefulPartitionedCall:output:0stack_1_block0_mb_dw_bn_44842stack_1_block0_mb_dw_bn_44844stack_1_block0_mb_dw_bn_44846stack_1_block0_mb_dw_bn_44848*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_43889?
*stack_1_block0_MB_dw_swish/PartitionedCallPartitionedCall8stack_1_block0_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_43004}
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_1/MeanMean3stack_1_block0_MB_dw_swish/PartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
0stack_1_block0_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_1/Mean:output:0stack_1_block0_se_1_conv_44854stack_1_block0_se_1_conv_44856*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_1_conv_layer_call_and_return_conditional_losses_43018?
'stack_1_block0_se_swish/PartitionedCallPartitionedCall9stack_1_block0_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_43034?
0stack_1_block0_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_1_block0_se_swish/PartitionedCall:output:0stack_1_block0_se_2_conv_44860stack_1_block0_se_2_conv_44862*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_2_conv_layer_call_and_return_conditional_losses_43046?
)stack_1_block0_se_sigmoid/PartitionedCallPartitionedCall9stack_1_block0_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_se_sigmoid_layer_call_and_return_conditional_losses_43057?
%stack_1_block0_se_out/PartitionedCallPartitionedCall3stack_1_block0_MB_dw_swish/PartitionedCall:output:02stack_1_block0_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_se_out_layer_call_and_return_conditional_losses_43065?
1stack_1_block0_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block0_se_out/PartitionedCall:output:0stack_1_block0_mb_pw_conv_44867*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_MB_pw_conv_layer_call_and_return_conditional_losses_43074?
/stack_1_block0_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_1_block0_MB_pw_conv/StatefulPartitionedCall:output:0stack_1_block0_mb_pw_bn_44870stack_1_block0_mb_pw_bn_44872stack_1_block0_mb_pw_bn_44874stack_1_block0_mb_pw_bn_44876*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43792?
%stack_1_block0_output/PartitionedCallPartitionedCall8stack_1_block0_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_output_layer_call_and_return_conditional_losses_43109?
-stack_1_block1_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block0_output/PartitionedCall:output:0stack_1_block1_mb_dw__44880*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_MB_dw__layer_call_and_return_conditional_losses_43120?
/stack_1_block1_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_1_block1_MB_dw_/StatefulPartitionedCall:output:0stack_1_block1_mb_dw_bn_44883stack_1_block1_mb_dw_bn_44885stack_1_block1_mb_dw_bn_44887stack_1_block1_mb_dw_bn_44889*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_43734?
*stack_1_block1_MB_dw_swish/PartitionedCallPartitionedCall8stack_1_block1_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_43161}
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_2/MeanMean3stack_1_block1_MB_dw_swish/PartitionedCall:output:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
0stack_1_block1_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_2/Mean:output:0stack_1_block1_se_1_conv_44895stack_1_block1_se_1_conv_44897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_1_conv_layer_call_and_return_conditional_losses_43175?
'stack_1_block1_se_swish/PartitionedCallPartitionedCall9stack_1_block1_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_43191?
0stack_1_block1_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_1_block1_se_swish/PartitionedCall:output:0stack_1_block1_se_2_conv_44901stack_1_block1_se_2_conv_44903*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_2_conv_layer_call_and_return_conditional_losses_43203?
)stack_1_block1_se_sigmoid/PartitionedCallPartitionedCall9stack_1_block1_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_se_sigmoid_layer_call_and_return_conditional_losses_43214?
%stack_1_block1_se_out/PartitionedCallPartitionedCall3stack_1_block1_MB_dw_swish/PartitionedCall:output:02stack_1_block1_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_se_out_layer_call_and_return_conditional_losses_43222?
1stack_1_block1_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block1_se_out/PartitionedCall:output:0stack_1_block1_mb_pw_conv_44908*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_MB_pw_conv_layer_call_and_return_conditional_losses_43231?
/stack_1_block1_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_1_block1_MB_pw_conv/StatefulPartitionedCall:output:0stack_1_block1_mb_pw_bn_44911stack_1_block1_mb_pw_bn_44913stack_1_block1_mb_pw_bn_44915stack_1_block1_mb_pw_bn_44917*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_43637?
/stack_1_block1_dropdrop/StatefulPartitionedCallStatefulPartitionedCall8stack_1_block1_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_43601?
%stack_1_block1_output/PartitionedCallPartitionedCall.stack_1_block0_output/PartitionedCall:output:08stack_1_block1_dropdrop/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_output_layer_call_and_return_conditional_losses_43275?
!post_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block1_output/PartitionedCall:output:0post_conv_44922*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_post_conv_layer_call_and_return_conditional_losses_43284?
post_bn/StatefulPartitionedCallStatefulPartitionedCall*post_conv/StatefulPartitionedCall:output:0post_bn_44925post_bn_44927post_bn_44929post_bn_44931*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_post_bn_layer_call_and_return_conditional_losses_43547?
post_swish/PartitionedCallPartitionedCall(post_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_post_swish_layer_call_and_return_conditional_losses_43325?
avg_pool/PartitionedCallPartitionedCall#post_swish/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_43332?
!head_drop/StatefulPartitionedCallStatefulPartitionedCall!avg_pool/PartitionedCall:output:00^stack_1_block1_dropdrop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_head_drop_layer_call_and_return_conditional_losses_43500?
#predictions/StatefulPartitionedCallStatefulPartitionedCall*head_drop/StatefulPartitionedCall:output:0predictions_44937predictions_44939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_43352{
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp"^head_drop/StatefulPartitionedCall ^post_bn/StatefulPartitionedCall"^post_conv/StatefulPartitionedCall$^predictions/StatefulPartitionedCall.^stack_0_block0_MB_dw_/StatefulPartitionedCall0^stack_0_block0_MB_dw_bn/StatefulPartitionedCall0^stack_0_block0_MB_pw_bn/StatefulPartitionedCall2^stack_0_block0_MB_pw_conv/StatefulPartitionedCall1^stack_0_block0_se_1_conv/StatefulPartitionedCall1^stack_0_block0_se_2_conv/StatefulPartitionedCall.^stack_1_block0_MB_dw_/StatefulPartitionedCall0^stack_1_block0_MB_dw_bn/StatefulPartitionedCall0^stack_1_block0_MB_pw_bn/StatefulPartitionedCall2^stack_1_block0_MB_pw_conv/StatefulPartitionedCall1^stack_1_block0_se_1_conv/StatefulPartitionedCall1^stack_1_block0_se_2_conv/StatefulPartitionedCall.^stack_1_block1_MB_dw_/StatefulPartitionedCall0^stack_1_block1_MB_dw_bn/StatefulPartitionedCall0^stack_1_block1_MB_pw_bn/StatefulPartitionedCall2^stack_1_block1_MB_pw_conv/StatefulPartitionedCall0^stack_1_block1_dropdrop/StatefulPartitionedCall1^stack_1_block1_se_1_conv/StatefulPartitionedCall1^stack_1_block1_se_2_conv/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!head_drop/StatefulPartitionedCall!head_drop/StatefulPartitionedCall2B
post_bn/StatefulPartitionedCallpost_bn/StatefulPartitionedCall2F
!post_conv/StatefulPartitionedCall!post_conv/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall2^
-stack_0_block0_MB_dw_/StatefulPartitionedCall-stack_0_block0_MB_dw_/StatefulPartitionedCall2b
/stack_0_block0_MB_dw_bn/StatefulPartitionedCall/stack_0_block0_MB_dw_bn/StatefulPartitionedCall2b
/stack_0_block0_MB_pw_bn/StatefulPartitionedCall/stack_0_block0_MB_pw_bn/StatefulPartitionedCall2f
1stack_0_block0_MB_pw_conv/StatefulPartitionedCall1stack_0_block0_MB_pw_conv/StatefulPartitionedCall2d
0stack_0_block0_se_1_conv/StatefulPartitionedCall0stack_0_block0_se_1_conv/StatefulPartitionedCall2d
0stack_0_block0_se_2_conv/StatefulPartitionedCall0stack_0_block0_se_2_conv/StatefulPartitionedCall2^
-stack_1_block0_MB_dw_/StatefulPartitionedCall-stack_1_block0_MB_dw_/StatefulPartitionedCall2b
/stack_1_block0_MB_dw_bn/StatefulPartitionedCall/stack_1_block0_MB_dw_bn/StatefulPartitionedCall2b
/stack_1_block0_MB_pw_bn/StatefulPartitionedCall/stack_1_block0_MB_pw_bn/StatefulPartitionedCall2f
1stack_1_block0_MB_pw_conv/StatefulPartitionedCall1stack_1_block0_MB_pw_conv/StatefulPartitionedCall2d
0stack_1_block0_se_1_conv/StatefulPartitionedCall0stack_1_block0_se_1_conv/StatefulPartitionedCall2d
0stack_1_block0_se_2_conv/StatefulPartitionedCall0stack_1_block0_se_2_conv/StatefulPartitionedCall2^
-stack_1_block1_MB_dw_/StatefulPartitionedCall-stack_1_block1_MB_dw_/StatefulPartitionedCall2b
/stack_1_block1_MB_dw_bn/StatefulPartitionedCall/stack_1_block1_MB_dw_bn/StatefulPartitionedCall2b
/stack_1_block1_MB_pw_bn/StatefulPartitionedCall/stack_1_block1_MB_pw_bn/StatefulPartitionedCall2f
1stack_1_block1_MB_pw_conv/StatefulPartitionedCall1stack_1_block1_MB_pw_conv/StatefulPartitionedCall2b
/stack_1_block1_dropdrop/StatefulPartitionedCall/stack_1_block1_dropdrop/StatefulPartitionedCall2d
0stack_1_block1_se_1_conv/StatefulPartitionedCall0stack_1_block1_se_1_conv/StatefulPartitionedCall2d
0stack_1_block1_se_2_conv/StatefulPartitionedCall0stack_1_block1_se_2_conv/StatefulPartitionedCall2B
stem_bn/StatefulPartitionedCallstem_bn/StatefulPartitionedCall2F
!stem_conv/StatefulPartitionedCall!stem_conv/StatefulPartitionedCall:Y U
0
_output_shapes
:????????? ?
!
_user_specified_name	input_1
?
?
'__inference_stem_bn_layer_call_fn_45860

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_42268?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
G__inference_EfficientNet_layer_call_and_return_conditional_losses_44397

inputs)
stem_conv_44239: 
stem_bn_44242: 
stem_bn_44244: 
stem_bn_44246: 
stem_bn_44248: 5
stack_0_block0_mb_dw__44252: +
stack_0_block0_mb_dw_bn_44255: +
stack_0_block0_mb_dw_bn_44257: +
stack_0_block0_mb_dw_bn_44259: +
stack_0_block0_mb_dw_bn_44261: 8
stack_0_block0_se_1_conv_44267: ,
stack_0_block0_se_1_conv_44269:8
stack_0_block0_se_2_conv_44273: ,
stack_0_block0_se_2_conv_44275: 9
stack_0_block0_mb_pw_conv_44280: +
stack_0_block0_mb_pw_bn_44283:+
stack_0_block0_mb_pw_bn_44285:+
stack_0_block0_mb_pw_bn_44287:+
stack_0_block0_mb_pw_bn_44289:5
stack_1_block0_mb_dw__44293:+
stack_1_block0_mb_dw_bn_44296:+
stack_1_block0_mb_dw_bn_44298:+
stack_1_block0_mb_dw_bn_44300:+
stack_1_block0_mb_dw_bn_44302:8
stack_1_block0_se_1_conv_44308:,
stack_1_block0_se_1_conv_44310:8
stack_1_block0_se_2_conv_44314:,
stack_1_block0_se_2_conv_44316:9
stack_1_block0_mb_pw_conv_44321:+
stack_1_block0_mb_pw_bn_44324:+
stack_1_block0_mb_pw_bn_44326:+
stack_1_block0_mb_pw_bn_44328:+
stack_1_block0_mb_pw_bn_44330:5
stack_1_block1_mb_dw__44334:+
stack_1_block1_mb_dw_bn_44337:+
stack_1_block1_mb_dw_bn_44339:+
stack_1_block1_mb_dw_bn_44341:+
stack_1_block1_mb_dw_bn_44343:8
stack_1_block1_se_1_conv_44349:,
stack_1_block1_se_1_conv_44351:8
stack_1_block1_se_2_conv_44355:,
stack_1_block1_se_2_conv_44357:9
stack_1_block1_mb_pw_conv_44362:+
stack_1_block1_mb_pw_bn_44365:+
stack_1_block1_mb_pw_bn_44367:+
stack_1_block1_mb_pw_bn_44369:+
stack_1_block1_mb_pw_bn_44371:*
post_conv_44376:?

post_bn_44379:	?

post_bn_44381:	?

post_bn_44383:	?

post_bn_44385:	?
$
predictions_44391:	?

predictions_44393:
identity??!head_drop/StatefulPartitionedCall?post_bn/StatefulPartitionedCall?!post_conv/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?-stack_0_block0_MB_dw_/StatefulPartitionedCall?/stack_0_block0_MB_dw_bn/StatefulPartitionedCall?/stack_0_block0_MB_pw_bn/StatefulPartitionedCall?1stack_0_block0_MB_pw_conv/StatefulPartitionedCall?0stack_0_block0_se_1_conv/StatefulPartitionedCall?0stack_0_block0_se_2_conv/StatefulPartitionedCall?-stack_1_block0_MB_dw_/StatefulPartitionedCall?/stack_1_block0_MB_dw_bn/StatefulPartitionedCall?/stack_1_block0_MB_pw_bn/StatefulPartitionedCall?1stack_1_block0_MB_pw_conv/StatefulPartitionedCall?0stack_1_block0_se_1_conv/StatefulPartitionedCall?0stack_1_block0_se_2_conv/StatefulPartitionedCall?-stack_1_block1_MB_dw_/StatefulPartitionedCall?/stack_1_block1_MB_dw_bn/StatefulPartitionedCall?/stack_1_block1_MB_pw_bn/StatefulPartitionedCall?1stack_1_block1_MB_pw_conv/StatefulPartitionedCall?/stack_1_block1_dropdrop/StatefulPartitionedCall?0stack_1_block1_se_1_conv/StatefulPartitionedCall?0stack_1_block1_se_2_conv/StatefulPartitionedCall?stem_bn/StatefulPartitionedCall?!stem_conv/StatefulPartitionedCall?
!stem_conv/StatefulPartitionedCallStatefulPartitionedCallinputsstem_conv_44239*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_42754?
stem_bn/StatefulPartitionedCallStatefulPartitionedCall*stem_conv/StatefulPartitionedCall:output:0stem_bn_44242stem_bn_44244stem_bn_44246stem_bn_44248*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_44102?
stem_swish/PartitionedCallPartitionedCall(stem_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_stem_swish_layer_call_and_return_conditional_losses_42795?
-stack_0_block0_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall#stem_swish/PartitionedCall:output:0stack_0_block0_mb_dw__44252*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_MB_dw__layer_call_and_return_conditional_losses_42806?
/stack_0_block0_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_0_block0_MB_dw_/StatefulPartitionedCall:output:0stack_0_block0_mb_dw_bn_44255stack_0_block0_mb_dw_bn_44257stack_0_block0_mb_dw_bn_44259stack_0_block0_mb_dw_bn_44261*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_44044?
*stack_0_block0_MB_dw_swish/PartitionedCallPartitionedCall8stack_0_block0_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_42847{
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean/MeanMean3stack_0_block0_MB_dw_swish/PartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(?
0stack_0_block0_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_mean/Mean:output:0stack_0_block0_se_1_conv_44267stack_0_block0_se_1_conv_44269*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_1_conv_layer_call_and_return_conditional_losses_42861?
'stack_0_block0_se_swish/PartitionedCallPartitionedCall9stack_0_block0_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_42877?
0stack_0_block0_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_0_block0_se_swish/PartitionedCall:output:0stack_0_block0_se_2_conv_44273stack_0_block0_se_2_conv_44275*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_2_conv_layer_call_and_return_conditional_losses_42889?
)stack_0_block0_se_sigmoid/PartitionedCallPartitionedCall9stack_0_block0_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_se_sigmoid_layer_call_and_return_conditional_losses_42900?
%stack_0_block0_se_out/PartitionedCallPartitionedCall3stack_0_block0_MB_dw_swish/PartitionedCall:output:02stack_0_block0_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_se_out_layer_call_and_return_conditional_losses_42908?
1stack_0_block0_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_0_block0_se_out/PartitionedCall:output:0stack_0_block0_mb_pw_conv_44280*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_MB_pw_conv_layer_call_and_return_conditional_losses_42917?
/stack_0_block0_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_0_block0_MB_pw_conv/StatefulPartitionedCall:output:0stack_0_block0_mb_pw_bn_44283stack_0_block0_mb_pw_bn_44285stack_0_block0_mb_pw_bn_44287stack_0_block0_mb_pw_bn_44289*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43947?
%stack_0_block0_output/PartitionedCallPartitionedCall8stack_0_block0_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_output_layer_call_and_return_conditional_losses_42952?
-stack_1_block0_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall.stack_0_block0_output/PartitionedCall:output:0stack_1_block0_mb_dw__44293*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_MB_dw__layer_call_and_return_conditional_losses_42963?
/stack_1_block0_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_1_block0_MB_dw_/StatefulPartitionedCall:output:0stack_1_block0_mb_dw_bn_44296stack_1_block0_mb_dw_bn_44298stack_1_block0_mb_dw_bn_44300stack_1_block0_mb_dw_bn_44302*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_43889?
*stack_1_block0_MB_dw_swish/PartitionedCallPartitionedCall8stack_1_block0_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_43004}
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_1/MeanMean3stack_1_block0_MB_dw_swish/PartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
0stack_1_block0_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_1/Mean:output:0stack_1_block0_se_1_conv_44308stack_1_block0_se_1_conv_44310*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_1_conv_layer_call_and_return_conditional_losses_43018?
'stack_1_block0_se_swish/PartitionedCallPartitionedCall9stack_1_block0_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_43034?
0stack_1_block0_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_1_block0_se_swish/PartitionedCall:output:0stack_1_block0_se_2_conv_44314stack_1_block0_se_2_conv_44316*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_2_conv_layer_call_and_return_conditional_losses_43046?
)stack_1_block0_se_sigmoid/PartitionedCallPartitionedCall9stack_1_block0_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_se_sigmoid_layer_call_and_return_conditional_losses_43057?
%stack_1_block0_se_out/PartitionedCallPartitionedCall3stack_1_block0_MB_dw_swish/PartitionedCall:output:02stack_1_block0_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_se_out_layer_call_and_return_conditional_losses_43065?
1stack_1_block0_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block0_se_out/PartitionedCall:output:0stack_1_block0_mb_pw_conv_44321*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_MB_pw_conv_layer_call_and_return_conditional_losses_43074?
/stack_1_block0_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_1_block0_MB_pw_conv/StatefulPartitionedCall:output:0stack_1_block0_mb_pw_bn_44324stack_1_block0_mb_pw_bn_44326stack_1_block0_mb_pw_bn_44328stack_1_block0_mb_pw_bn_44330*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43792?
%stack_1_block0_output/PartitionedCallPartitionedCall8stack_1_block0_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_output_layer_call_and_return_conditional_losses_43109?
-stack_1_block1_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block0_output/PartitionedCall:output:0stack_1_block1_mb_dw__44334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_MB_dw__layer_call_and_return_conditional_losses_43120?
/stack_1_block1_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_1_block1_MB_dw_/StatefulPartitionedCall:output:0stack_1_block1_mb_dw_bn_44337stack_1_block1_mb_dw_bn_44339stack_1_block1_mb_dw_bn_44341stack_1_block1_mb_dw_bn_44343*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_43734?
*stack_1_block1_MB_dw_swish/PartitionedCallPartitionedCall8stack_1_block1_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_43161}
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_2/MeanMean3stack_1_block1_MB_dw_swish/PartitionedCall:output:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
0stack_1_block1_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_2/Mean:output:0stack_1_block1_se_1_conv_44349stack_1_block1_se_1_conv_44351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_1_conv_layer_call_and_return_conditional_losses_43175?
'stack_1_block1_se_swish/PartitionedCallPartitionedCall9stack_1_block1_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_43191?
0stack_1_block1_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_1_block1_se_swish/PartitionedCall:output:0stack_1_block1_se_2_conv_44355stack_1_block1_se_2_conv_44357*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_2_conv_layer_call_and_return_conditional_losses_43203?
)stack_1_block1_se_sigmoid/PartitionedCallPartitionedCall9stack_1_block1_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_se_sigmoid_layer_call_and_return_conditional_losses_43214?
%stack_1_block1_se_out/PartitionedCallPartitionedCall3stack_1_block1_MB_dw_swish/PartitionedCall:output:02stack_1_block1_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_se_out_layer_call_and_return_conditional_losses_43222?
1stack_1_block1_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block1_se_out/PartitionedCall:output:0stack_1_block1_mb_pw_conv_44362*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_MB_pw_conv_layer_call_and_return_conditional_losses_43231?
/stack_1_block1_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_1_block1_MB_pw_conv/StatefulPartitionedCall:output:0stack_1_block1_mb_pw_bn_44365stack_1_block1_mb_pw_bn_44367stack_1_block1_mb_pw_bn_44369stack_1_block1_mb_pw_bn_44371*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_43637?
/stack_1_block1_dropdrop/StatefulPartitionedCallStatefulPartitionedCall8stack_1_block1_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_43601?
%stack_1_block1_output/PartitionedCallPartitionedCall.stack_1_block0_output/PartitionedCall:output:08stack_1_block1_dropdrop/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_output_layer_call_and_return_conditional_losses_43275?
!post_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block1_output/PartitionedCall:output:0post_conv_44376*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_post_conv_layer_call_and_return_conditional_losses_43284?
post_bn/StatefulPartitionedCallStatefulPartitionedCall*post_conv/StatefulPartitionedCall:output:0post_bn_44379post_bn_44381post_bn_44383post_bn_44385*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_post_bn_layer_call_and_return_conditional_losses_43547?
post_swish/PartitionedCallPartitionedCall(post_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_post_swish_layer_call_and_return_conditional_losses_43325?
avg_pool/PartitionedCallPartitionedCall#post_swish/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_43332?
!head_drop/StatefulPartitionedCallStatefulPartitionedCall!avg_pool/PartitionedCall:output:00^stack_1_block1_dropdrop/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_head_drop_layer_call_and_return_conditional_losses_43500?
#predictions/StatefulPartitionedCallStatefulPartitionedCall*head_drop/StatefulPartitionedCall:output:0predictions_44391predictions_44393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_43352{
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp"^head_drop/StatefulPartitionedCall ^post_bn/StatefulPartitionedCall"^post_conv/StatefulPartitionedCall$^predictions/StatefulPartitionedCall.^stack_0_block0_MB_dw_/StatefulPartitionedCall0^stack_0_block0_MB_dw_bn/StatefulPartitionedCall0^stack_0_block0_MB_pw_bn/StatefulPartitionedCall2^stack_0_block0_MB_pw_conv/StatefulPartitionedCall1^stack_0_block0_se_1_conv/StatefulPartitionedCall1^stack_0_block0_se_2_conv/StatefulPartitionedCall.^stack_1_block0_MB_dw_/StatefulPartitionedCall0^stack_1_block0_MB_dw_bn/StatefulPartitionedCall0^stack_1_block0_MB_pw_bn/StatefulPartitionedCall2^stack_1_block0_MB_pw_conv/StatefulPartitionedCall1^stack_1_block0_se_1_conv/StatefulPartitionedCall1^stack_1_block0_se_2_conv/StatefulPartitionedCall.^stack_1_block1_MB_dw_/StatefulPartitionedCall0^stack_1_block1_MB_dw_bn/StatefulPartitionedCall0^stack_1_block1_MB_pw_bn/StatefulPartitionedCall2^stack_1_block1_MB_pw_conv/StatefulPartitionedCall0^stack_1_block1_dropdrop/StatefulPartitionedCall1^stack_1_block1_se_1_conv/StatefulPartitionedCall1^stack_1_block1_se_2_conv/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!head_drop/StatefulPartitionedCall!head_drop/StatefulPartitionedCall2B
post_bn/StatefulPartitionedCallpost_bn/StatefulPartitionedCall2F
!post_conv/StatefulPartitionedCall!post_conv/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall2^
-stack_0_block0_MB_dw_/StatefulPartitionedCall-stack_0_block0_MB_dw_/StatefulPartitionedCall2b
/stack_0_block0_MB_dw_bn/StatefulPartitionedCall/stack_0_block0_MB_dw_bn/StatefulPartitionedCall2b
/stack_0_block0_MB_pw_bn/StatefulPartitionedCall/stack_0_block0_MB_pw_bn/StatefulPartitionedCall2f
1stack_0_block0_MB_pw_conv/StatefulPartitionedCall1stack_0_block0_MB_pw_conv/StatefulPartitionedCall2d
0stack_0_block0_se_1_conv/StatefulPartitionedCall0stack_0_block0_se_1_conv/StatefulPartitionedCall2d
0stack_0_block0_se_2_conv/StatefulPartitionedCall0stack_0_block0_se_2_conv/StatefulPartitionedCall2^
-stack_1_block0_MB_dw_/StatefulPartitionedCall-stack_1_block0_MB_dw_/StatefulPartitionedCall2b
/stack_1_block0_MB_dw_bn/StatefulPartitionedCall/stack_1_block0_MB_dw_bn/StatefulPartitionedCall2b
/stack_1_block0_MB_pw_bn/StatefulPartitionedCall/stack_1_block0_MB_pw_bn/StatefulPartitionedCall2f
1stack_1_block0_MB_pw_conv/StatefulPartitionedCall1stack_1_block0_MB_pw_conv/StatefulPartitionedCall2d
0stack_1_block0_se_1_conv/StatefulPartitionedCall0stack_1_block0_se_1_conv/StatefulPartitionedCall2d
0stack_1_block0_se_2_conv/StatefulPartitionedCall0stack_1_block0_se_2_conv/StatefulPartitionedCall2^
-stack_1_block1_MB_dw_/StatefulPartitionedCall-stack_1_block1_MB_dw_/StatefulPartitionedCall2b
/stack_1_block1_MB_dw_bn/StatefulPartitionedCall/stack_1_block1_MB_dw_bn/StatefulPartitionedCall2b
/stack_1_block1_MB_pw_bn/StatefulPartitionedCall/stack_1_block1_MB_pw_bn/StatefulPartitionedCall2f
1stack_1_block1_MB_pw_conv/StatefulPartitionedCall1stack_1_block1_MB_pw_conv/StatefulPartitionedCall2b
/stack_1_block1_dropdrop/StatefulPartitionedCall/stack_1_block1_dropdrop/StatefulPartitionedCall2d
0stack_1_block1_se_1_conv/StatefulPartitionedCall0stack_1_block1_se_1_conv/StatefulPartitionedCall2d
0stack_1_block1_se_2_conv/StatefulPartitionedCall0stack_1_block1_se_2_conv/StatefulPartitionedCall2B
stem_bn/StatefulPartitionedCallstem_bn/StatefulPartitionedCall2F
!stem_conv/StatefulPartitionedCall!stem_conv/StatefulPartitionedCall:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
?
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46028

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42827x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
B__inference_post_bn_layer_call_and_return_conditional_losses_47280

inputs&
readvariableop_resource:	?
(
readvariableop_1_resource:	?
7
(fusedbatchnormv3_readvariableop_resource:	?
9
*fusedbatchnormv3_readvariableop_1_resource:	?

identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?
*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????`?
:?
:?
:?
:?
:*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????`?
?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????`?
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?	
?
P__inference_stack_1_block1_MB_dw__layer_call_and_return_conditional_losses_46743

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
i
IdentityIdentitydepthwise:output:0^NoOp*
T0*/
_output_shapes
:?????????`a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?	
c
D__inference_head_drop_layer_call_and_return_conditional_losses_43500

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????
C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????
*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????
p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????
j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????
Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
9__inference_stack_1_block0_MB_pw_conv_layer_call_fn_46587

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_MB_pw_conv_layer_call_and_return_conditional_losses_43074w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_47010

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_43252w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42460

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_post_bn_layer_call_and_return_conditional_losses_43547

inputs&
readvariableop_resource:	?
(
readvariableop_1_resource:	?
7
(fusedbatchnormv3_readvariableop_resource:	?
9
*fusedbatchnormv3_readvariableop_1_resource:	?

identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?
*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????`?
:?
:?
:?
:?
:*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????`?
?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????`?
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?
S
7__inference_stack_1_block0_se_swish_layer_call_fn_46529

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_43034h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42332

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
9__inference_stack_0_block0_MB_pw_conv_layer_call_fn_46210

inputs!
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_MB_pw_conv_layer_call_and_return_conditional_losses_42917x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????? : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
a
5__inference_stack_1_block1_se_out_layer_call_fn_46951
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_se_out_layer_call_and_return_conditional_losses_43222h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????`:?????????:Y U
/
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
p
7__inference_stack_1_block1_dropdrop_layer_call_fn_47105

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_43601w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47766
result_grads_0
result_grads_11
-sigmoid_efficientnet_post_bn_fusedbatchnormv3
identity?
SigmoidSigmoid-sigmoid_efficientnet_post_bn_fusedbatchnormv3^result_grads_0*
T0*0
_output_shapes
:?????????`?
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????`?
}
mulMul-sigmoid_efficientnet_post_bn_fusedbatchnormv3sub:z:0*
T0*0
_output_shapes
:?????????`?
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????`?
]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????`?
b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????`?
Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????`?
"
identityIdentity:output:0*g
_input_shapesV
T:?????????`?
:?????????`?
:?????????`?
:` \
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????`?

?
D
(__inference_avg_pool_layer_call_fn_47300

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_42737i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46490

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46436

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
8__inference_stack_1_block0_se_2_conv_layer_call_fn_46548

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_2_conv_layer_call_and_return_conditional_losses_43046w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46700

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_47856
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?
?
B__inference_post_bn_layer_call_and_return_conditional_losses_47226

inputs&
readvariableop_resource:	?
(
readvariableop_1_resource:	?
7
(fusedbatchnormv3_readvariableop_resource:	?
9
*fusedbatchnormv3_readvariableop_1_resource:	?

identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?
*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????
:?
:?
:?
:?
:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????
?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????

 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_47886
result_grads_0
result_grads_1
sigmoid_inputs
identityn
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*0
_output_shapes
:?????????`?
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????`?
^
mulMulsigmoid_inputssub:z:0*
T0*0
_output_shapes
:?????????`?
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????`?
]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????`?
b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????`?
Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????`?
"
identityIdentity:output:0*g
_input_shapesV
T:?????????`?
:?????????`?
:?????????`?
:` \
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????`?

?

?
S__inference_stack_1_block0_se_2_conv_layer_call_and_return_conditional_losses_43046

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
V
:__inference_stack_1_block0_MB_dw_swish_layer_call_fn_46495

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_43004h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_42557

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_stem_conv_layer_call_and_return_conditional_losses_42754

inputs8
conv2d_readvariableop_resource: 
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????? ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? ?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
p
T__inference_stack_1_block1_se_sigmoid_layer_call_and_return_conditional_losses_43214

inputs
identityT
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????[
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_47781
result_grads_0
result_grads_1
sigmoid_inputs
identityn
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ^
mulMulsigmoid_inputssub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
?
?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_43637

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
p
T__inference_stack_0_block0_se_sigmoid_layer_call_and_return_conditional_losses_46191

inputs
identityT
SigmoidSigmoidinputs*
T0*/
_output_shapes
:????????? [
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_43734

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?	
?
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46230

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42365?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
|
P__inference_stack_1_block0_se_out_layer_call_and_return_conditional_losses_46580
inputs_0
inputs_1
identityX
mulMulinputs_0inputs_1*
T0*/
_output_shapes
:?????????`W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????`:?????????:Y U
/
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
??
?
G__inference_EfficientNet_layer_call_and_return_conditional_losses_43359

inputs)
stem_conv_42755: 
stem_bn_42776: 
stem_bn_42778: 
stem_bn_42780: 
stem_bn_42782: 5
stack_0_block0_mb_dw__42807: +
stack_0_block0_mb_dw_bn_42828: +
stack_0_block0_mb_dw_bn_42830: +
stack_0_block0_mb_dw_bn_42832: +
stack_0_block0_mb_dw_bn_42834: 8
stack_0_block0_se_1_conv_42862: ,
stack_0_block0_se_1_conv_42864:8
stack_0_block0_se_2_conv_42890: ,
stack_0_block0_se_2_conv_42892: 9
stack_0_block0_mb_pw_conv_42918: +
stack_0_block0_mb_pw_bn_42939:+
stack_0_block0_mb_pw_bn_42941:+
stack_0_block0_mb_pw_bn_42943:+
stack_0_block0_mb_pw_bn_42945:5
stack_1_block0_mb_dw__42964:+
stack_1_block0_mb_dw_bn_42985:+
stack_1_block0_mb_dw_bn_42987:+
stack_1_block0_mb_dw_bn_42989:+
stack_1_block0_mb_dw_bn_42991:8
stack_1_block0_se_1_conv_43019:,
stack_1_block0_se_1_conv_43021:8
stack_1_block0_se_2_conv_43047:,
stack_1_block0_se_2_conv_43049:9
stack_1_block0_mb_pw_conv_43075:+
stack_1_block0_mb_pw_bn_43096:+
stack_1_block0_mb_pw_bn_43098:+
stack_1_block0_mb_pw_bn_43100:+
stack_1_block0_mb_pw_bn_43102:5
stack_1_block1_mb_dw__43121:+
stack_1_block1_mb_dw_bn_43142:+
stack_1_block1_mb_dw_bn_43144:+
stack_1_block1_mb_dw_bn_43146:+
stack_1_block1_mb_dw_bn_43148:8
stack_1_block1_se_1_conv_43176:,
stack_1_block1_se_1_conv_43178:8
stack_1_block1_se_2_conv_43204:,
stack_1_block1_se_2_conv_43206:9
stack_1_block1_mb_pw_conv_43232:+
stack_1_block1_mb_pw_bn_43253:+
stack_1_block1_mb_pw_bn_43255:+
stack_1_block1_mb_pw_bn_43257:+
stack_1_block1_mb_pw_bn_43259:*
post_conv_43285:?

post_bn_43306:	?

post_bn_43308:	?

post_bn_43310:	?

post_bn_43312:	?
$
predictions_43353:	?

predictions_43355:
identity??post_bn/StatefulPartitionedCall?!post_conv/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?-stack_0_block0_MB_dw_/StatefulPartitionedCall?/stack_0_block0_MB_dw_bn/StatefulPartitionedCall?/stack_0_block0_MB_pw_bn/StatefulPartitionedCall?1stack_0_block0_MB_pw_conv/StatefulPartitionedCall?0stack_0_block0_se_1_conv/StatefulPartitionedCall?0stack_0_block0_se_2_conv/StatefulPartitionedCall?-stack_1_block0_MB_dw_/StatefulPartitionedCall?/stack_1_block0_MB_dw_bn/StatefulPartitionedCall?/stack_1_block0_MB_pw_bn/StatefulPartitionedCall?1stack_1_block0_MB_pw_conv/StatefulPartitionedCall?0stack_1_block0_se_1_conv/StatefulPartitionedCall?0stack_1_block0_se_2_conv/StatefulPartitionedCall?-stack_1_block1_MB_dw_/StatefulPartitionedCall?/stack_1_block1_MB_dw_bn/StatefulPartitionedCall?/stack_1_block1_MB_pw_bn/StatefulPartitionedCall?1stack_1_block1_MB_pw_conv/StatefulPartitionedCall?0stack_1_block1_se_1_conv/StatefulPartitionedCall?0stack_1_block1_se_2_conv/StatefulPartitionedCall?stem_bn/StatefulPartitionedCall?!stem_conv/StatefulPartitionedCall?
!stem_conv/StatefulPartitionedCallStatefulPartitionedCallinputsstem_conv_42755*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_42754?
stem_bn/StatefulPartitionedCallStatefulPartitionedCall*stem_conv/StatefulPartitionedCall:output:0stem_bn_42776stem_bn_42778stem_bn_42780stem_bn_42782*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_42775?
stem_swish/PartitionedCallPartitionedCall(stem_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_stem_swish_layer_call_and_return_conditional_losses_42795?
-stack_0_block0_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall#stem_swish/PartitionedCall:output:0stack_0_block0_mb_dw__42807*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_MB_dw__layer_call_and_return_conditional_losses_42806?
/stack_0_block0_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_0_block0_MB_dw_/StatefulPartitionedCall:output:0stack_0_block0_mb_dw_bn_42828stack_0_block0_mb_dw_bn_42830stack_0_block0_mb_dw_bn_42832stack_0_block0_mb_dw_bn_42834*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42827?
*stack_0_block0_MB_dw_swish/PartitionedCallPartitionedCall8stack_0_block0_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_42847{
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean/MeanMean3stack_0_block0_MB_dw_swish/PartitionedCall:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(?
0stack_0_block0_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_mean/Mean:output:0stack_0_block0_se_1_conv_42862stack_0_block0_se_1_conv_42864*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_1_conv_layer_call_and_return_conditional_losses_42861?
'stack_0_block0_se_swish/PartitionedCallPartitionedCall9stack_0_block0_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_42877?
0stack_0_block0_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_0_block0_se_swish/PartitionedCall:output:0stack_0_block0_se_2_conv_42890stack_0_block0_se_2_conv_42892*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_2_conv_layer_call_and_return_conditional_losses_42889?
)stack_0_block0_se_sigmoid/PartitionedCallPartitionedCall9stack_0_block0_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_se_sigmoid_layer_call_and_return_conditional_losses_42900?
%stack_0_block0_se_out/PartitionedCallPartitionedCall3stack_0_block0_MB_dw_swish/PartitionedCall:output:02stack_0_block0_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_se_out_layer_call_and_return_conditional_losses_42908?
1stack_0_block0_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_0_block0_se_out/PartitionedCall:output:0stack_0_block0_mb_pw_conv_42918*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_0_block0_MB_pw_conv_layer_call_and_return_conditional_losses_42917?
/stack_0_block0_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_0_block0_MB_pw_conv/StatefulPartitionedCall:output:0stack_0_block0_mb_pw_bn_42939stack_0_block0_mb_pw_bn_42941stack_0_block0_mb_pw_bn_42943stack_0_block0_mb_pw_bn_42945*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42938?
%stack_0_block0_output/PartitionedCallPartitionedCall8stack_0_block0_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_output_layer_call_and_return_conditional_losses_42952?
-stack_1_block0_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall.stack_0_block0_output/PartitionedCall:output:0stack_1_block0_mb_dw__42964*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_MB_dw__layer_call_and_return_conditional_losses_42963?
/stack_1_block0_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_1_block0_MB_dw_/StatefulPartitionedCall:output:0stack_1_block0_mb_dw_bn_42985stack_1_block0_mb_dw_bn_42987stack_1_block0_mb_dw_bn_42989stack_1_block0_mb_dw_bn_42991*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42984?
*stack_1_block0_MB_dw_swish/PartitionedCallPartitionedCall8stack_1_block0_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_43004}
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_1/MeanMean3stack_1_block0_MB_dw_swish/PartitionedCall:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
0stack_1_block0_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_1/Mean:output:0stack_1_block0_se_1_conv_43019stack_1_block0_se_1_conv_43021*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_1_conv_layer_call_and_return_conditional_losses_43018?
'stack_1_block0_se_swish/PartitionedCallPartitionedCall9stack_1_block0_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_43034?
0stack_1_block0_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_1_block0_se_swish/PartitionedCall:output:0stack_1_block0_se_2_conv_43047stack_1_block0_se_2_conv_43049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_2_conv_layer_call_and_return_conditional_losses_43046?
)stack_1_block0_se_sigmoid/PartitionedCallPartitionedCall9stack_1_block0_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_se_sigmoid_layer_call_and_return_conditional_losses_43057?
%stack_1_block0_se_out/PartitionedCallPartitionedCall3stack_1_block0_MB_dw_swish/PartitionedCall:output:02stack_1_block0_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_se_out_layer_call_and_return_conditional_losses_43065?
1stack_1_block0_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block0_se_out/PartitionedCall:output:0stack_1_block0_mb_pw_conv_43075*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_MB_pw_conv_layer_call_and_return_conditional_losses_43074?
/stack_1_block0_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_1_block0_MB_pw_conv/StatefulPartitionedCall:output:0stack_1_block0_mb_pw_bn_43096stack_1_block0_mb_pw_bn_43098stack_1_block0_mb_pw_bn_43100stack_1_block0_mb_pw_bn_43102*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43095?
%stack_1_block0_output/PartitionedCallPartitionedCall8stack_1_block0_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_output_layer_call_and_return_conditional_losses_43109?
-stack_1_block1_MB_dw_/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block0_output/PartitionedCall:output:0stack_1_block1_mb_dw__43121*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_MB_dw__layer_call_and_return_conditional_losses_43120?
/stack_1_block1_MB_dw_bn/StatefulPartitionedCallStatefulPartitionedCall6stack_1_block1_MB_dw_/StatefulPartitionedCall:output:0stack_1_block1_mb_dw_bn_43142stack_1_block1_mb_dw_bn_43144stack_1_block1_mb_dw_bn_43146stack_1_block1_mb_dw_bn_43148*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_43141?
*stack_1_block1_MB_dw_swish/PartitionedCallPartitionedCall8stack_1_block1_MB_dw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_43161}
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_2/MeanMean3stack_1_block1_MB_dw_swish/PartitionedCall:output:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
0stack_1_block1_se_1_conv/StatefulPartitionedCallStatefulPartitionedCall#tf.math.reduce_mean_2/Mean:output:0stack_1_block1_se_1_conv_43176stack_1_block1_se_1_conv_43178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_1_conv_layer_call_and_return_conditional_losses_43175?
'stack_1_block1_se_swish/PartitionedCallPartitionedCall9stack_1_block1_se_1_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_43191?
0stack_1_block1_se_2_conv/StatefulPartitionedCallStatefulPartitionedCall0stack_1_block1_se_swish/PartitionedCall:output:0stack_1_block1_se_2_conv_43204stack_1_block1_se_2_conv_43206*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_2_conv_layer_call_and_return_conditional_losses_43203?
)stack_1_block1_se_sigmoid/PartitionedCallPartitionedCall9stack_1_block1_se_2_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_se_sigmoid_layer_call_and_return_conditional_losses_43214?
%stack_1_block1_se_out/PartitionedCallPartitionedCall3stack_1_block1_MB_dw_swish/PartitionedCall:output:02stack_1_block1_se_sigmoid/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_se_out_layer_call_and_return_conditional_losses_43222?
1stack_1_block1_MB_pw_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block1_se_out/PartitionedCall:output:0stack_1_block1_mb_pw_conv_43232*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_MB_pw_conv_layer_call_and_return_conditional_losses_43231?
/stack_1_block1_MB_pw_bn/StatefulPartitionedCallStatefulPartitionedCall:stack_1_block1_MB_pw_conv/StatefulPartitionedCall:output:0stack_1_block1_mb_pw_bn_43253stack_1_block1_mb_pw_bn_43255stack_1_block1_mb_pw_bn_43257stack_1_block1_mb_pw_bn_43259*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_43252?
'stack_1_block1_dropdrop/PartitionedCallPartitionedCall8stack_1_block1_MB_pw_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_43267?
%stack_1_block1_output/PartitionedCallPartitionedCall.stack_1_block0_output/PartitionedCall:output:00stack_1_block1_dropdrop/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block1_output_layer_call_and_return_conditional_losses_43275?
!post_conv/StatefulPartitionedCallStatefulPartitionedCall.stack_1_block1_output/PartitionedCall:output:0post_conv_43285*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_post_conv_layer_call_and_return_conditional_losses_43284?
post_bn/StatefulPartitionedCallStatefulPartitionedCall*post_conv/StatefulPartitionedCall:output:0post_bn_43306post_bn_43308post_bn_43310post_bn_43312*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_post_bn_layer_call_and_return_conditional_losses_43305?
post_swish/PartitionedCallPartitionedCall(post_bn/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_post_swish_layer_call_and_return_conditional_losses_43325?
avg_pool/PartitionedCallPartitionedCall#post_swish/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_43332?
head_drop/PartitionedCallPartitionedCall!avg_pool/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_head_drop_layer_call_and_return_conditional_losses_43339?
#predictions/StatefulPartitionedCallStatefulPartitionedCall"head_drop/PartitionedCall:output:0predictions_43353predictions_43355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_43352{
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????	
NoOpNoOp ^post_bn/StatefulPartitionedCall"^post_conv/StatefulPartitionedCall$^predictions/StatefulPartitionedCall.^stack_0_block0_MB_dw_/StatefulPartitionedCall0^stack_0_block0_MB_dw_bn/StatefulPartitionedCall0^stack_0_block0_MB_pw_bn/StatefulPartitionedCall2^stack_0_block0_MB_pw_conv/StatefulPartitionedCall1^stack_0_block0_se_1_conv/StatefulPartitionedCall1^stack_0_block0_se_2_conv/StatefulPartitionedCall.^stack_1_block0_MB_dw_/StatefulPartitionedCall0^stack_1_block0_MB_dw_bn/StatefulPartitionedCall0^stack_1_block0_MB_pw_bn/StatefulPartitionedCall2^stack_1_block0_MB_pw_conv/StatefulPartitionedCall1^stack_1_block0_se_1_conv/StatefulPartitionedCall1^stack_1_block0_se_2_conv/StatefulPartitionedCall.^stack_1_block1_MB_dw_/StatefulPartitionedCall0^stack_1_block1_MB_dw_bn/StatefulPartitionedCall0^stack_1_block1_MB_pw_bn/StatefulPartitionedCall2^stack_1_block1_MB_pw_conv/StatefulPartitionedCall1^stack_1_block1_se_1_conv/StatefulPartitionedCall1^stack_1_block1_se_2_conv/StatefulPartitionedCall ^stem_bn/StatefulPartitionedCall"^stem_conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
post_bn/StatefulPartitionedCallpost_bn/StatefulPartitionedCall2F
!post_conv/StatefulPartitionedCall!post_conv/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall2^
-stack_0_block0_MB_dw_/StatefulPartitionedCall-stack_0_block0_MB_dw_/StatefulPartitionedCall2b
/stack_0_block0_MB_dw_bn/StatefulPartitionedCall/stack_0_block0_MB_dw_bn/StatefulPartitionedCall2b
/stack_0_block0_MB_pw_bn/StatefulPartitionedCall/stack_0_block0_MB_pw_bn/StatefulPartitionedCall2f
1stack_0_block0_MB_pw_conv/StatefulPartitionedCall1stack_0_block0_MB_pw_conv/StatefulPartitionedCall2d
0stack_0_block0_se_1_conv/StatefulPartitionedCall0stack_0_block0_se_1_conv/StatefulPartitionedCall2d
0stack_0_block0_se_2_conv/StatefulPartitionedCall0stack_0_block0_se_2_conv/StatefulPartitionedCall2^
-stack_1_block0_MB_dw_/StatefulPartitionedCall-stack_1_block0_MB_dw_/StatefulPartitionedCall2b
/stack_1_block0_MB_dw_bn/StatefulPartitionedCall/stack_1_block0_MB_dw_bn/StatefulPartitionedCall2b
/stack_1_block0_MB_pw_bn/StatefulPartitionedCall/stack_1_block0_MB_pw_bn/StatefulPartitionedCall2f
1stack_1_block0_MB_pw_conv/StatefulPartitionedCall1stack_1_block0_MB_pw_conv/StatefulPartitionedCall2d
0stack_1_block0_se_1_conv/StatefulPartitionedCall0stack_1_block0_se_1_conv/StatefulPartitionedCall2d
0stack_1_block0_se_2_conv/StatefulPartitionedCall0stack_1_block0_se_2_conv/StatefulPartitionedCall2^
-stack_1_block1_MB_dw_/StatefulPartitionedCall-stack_1_block1_MB_dw_/StatefulPartitionedCall2b
/stack_1_block1_MB_dw_bn/StatefulPartitionedCall/stack_1_block1_MB_dw_bn/StatefulPartitionedCall2b
/stack_1_block1_MB_pw_bn/StatefulPartitionedCall/stack_1_block1_MB_pw_bn/StatefulPartitionedCall2f
1stack_1_block1_MB_pw_conv/StatefulPartitionedCall1stack_1_block1_MB_pw_conv/StatefulPartitionedCall2d
0stack_1_block1_se_1_conv/StatefulPartitionedCall0stack_1_block1_se_1_conv/StatefulPartitionedCall2d
0stack_1_block1_se_2_conv/StatefulPartitionedCall0stack_1_block1_se_2_conv/StatefulPartitionedCall2B
stem_bn/StatefulPartitionedCallstem_bn/StatefulPartitionedCall2F
!stem_conv/StatefulPartitionedCall!stem_conv/StatefulPartitionedCall:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
?
8__inference_stack_0_block0_se_2_conv_layer_call_fn_46171

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_0_block0_se_2_conv_layer_call_and_return_conditional_losses_42889w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
B__inference_post_bn_layer_call_and_return_conditional_losses_42716

inputs&
readvariableop_resource:	?
(
readvariableop_1_resource:	?
7
(fusedbatchnormv3_readvariableop_resource:	?
9
*fusedbatchnormv3_readvariableop_1_resource:	?

identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?
*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????
:?
:?
:?
:?
:*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????
?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????

 
_user_specified_nameinputs
?
?
8__inference_stack_1_block1_se_1_conv_layer_call_fn_46891

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_1_conv_layer_call_and_return_conditional_losses_43175w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
b
D__inference_head_drop_layer_call_and_return_conditional_losses_47332

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????
\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
c
E__inference_post_swish_layer_call_and_return_conditional_losses_43325

inputs

identity_1U
SigmoidSigmoidinputs*
T0*0
_output_shapes
:?????????`?
Z
mulMulinputsSigmoid:y:0*
T0*0
_output_shapes
:?????????`?
X
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:?????????`?
?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-43318*L
_output_shapes:
8:?????????`?
:?????????`?
e

Identity_1IdentityIdentityN:output:0*
T0*0
_output_shapes
:?????????`?
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????`?
:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?
z
P__inference_stack_0_block0_se_out_layer_call_and_return_conditional_losses_42908

inputs
inputs_1
identityW
mulMulinputsinputs_1*
T0*0
_output_shapes
:?????????? X
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????? :????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs:WS
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_47826
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?
?
"__inference_internal_grad_fn_48036
result_grads_0
result_grads_14
0sigmoid_stack_0_block0_mb_dw_bn_fusedbatchnormv3
identity?
SigmoidSigmoid0sigmoid_stack_0_block0_mb_dw_bn_fusedbatchnormv3^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
mulMul0sigmoid_stack_0_block0_mb_dw_bn_fusedbatchnormv3sub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
?
S
7__inference_stack_1_block1_se_swish_layer_call_fn_46906

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_43191h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46305

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46113

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42938

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_43141

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

?
S__inference_stack_0_block0_se_1_conv_layer_call_and_return_conditional_losses_46147

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_stem_conv_layer_call_fn_45827

inputs!
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_stem_conv_layer_call_and_return_conditional_losses_42754x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:????????? ?: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
Q
5__inference_stack_1_block0_output_layer_call_fn_46723

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_1_block0_output_layer_call_and_return_conditional_losses_43109h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

?
S__inference_stack_1_block1_se_1_conv_layer_call_and_return_conditional_losses_46901

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46867

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
p
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_47110

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????`c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
D
(__inference_avg_pool_layer_call_fn_47305

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_avg_pool_layer_call_and_return_conditional_losses_43332a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????`?
:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46718

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
V
:__inference_stack_1_block1_MB_dw_swish_layer_call_fn_46872

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *^
fYRW
U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_43161h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?	
?
P__inference_stack_1_block0_MB_dw__layer_call_and_return_conditional_losses_46366

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
i
IdentityIdentitydepthwise:output:0^NoOp*
T0*/
_output_shapes
:?????????`a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_avg_pool_layer_call_and_return_conditional_losses_47311

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_stem_bn_layer_call_fn_45886

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_44102x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
p
R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_43191

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:??????????
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-43184*J
_output_shapes8
6:?????????:?????????d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47976
result_grads_0
result_grads_14
0sigmoid_stack_1_block1_mb_dw_bn_fusedbatchnormv3
identity?
SigmoidSigmoid0sigmoid_stack_1_block1_mb_dw_bn_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`
mulMul0sigmoid_stack_1_block1_mb_dw_bn_fusedbatchnormv3sub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?
Q
5__inference_stack_0_block0_output_layer_call_fn_46346

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_output_layer_call_and_return_conditional_losses_42952i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
P__inference_stack_1_block1_MB_dw__layer_call_and_return_conditional_losses_43120

inputs;
!depthwise_readvariableop_resource:
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
i
IdentityIdentitydepthwise:output:0^NoOp*
T0*/
_output_shapes
:?????????`a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

?
S__inference_stack_1_block1_se_1_conv_layer_call_and_return_conditional_losses_43175

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42827

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?	
?
P__inference_stack_0_block0_MB_dw__layer_call_and_return_conditional_losses_45989

inputs;
!depthwise_readvariableop_resource: 
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
j
IdentityIdentitydepthwise:output:0^NoOp*
T0*0
_output_shapes
:?????????? a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????? : 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
B__inference_post_bn_layer_call_and_return_conditional_losses_47244

inputs&
readvariableop_resource:	?
(
readvariableop_1_resource:	?
7
(fusedbatchnormv3_readvariableop_resource:	?
9
*fusedbatchnormv3_readvariableop_1_resource:	?

identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?
*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????
:?
:?
:?
:?
:*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????
?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????
: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????

 
_user_specified_nameinputs
?
?
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46269

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43947x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_47811
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?	
?
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_46997

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_42652?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_43252

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
,__inference_EfficientNet_layer_call_fn_43470
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: #
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: 

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:$

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:$

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:$

unknown_37:

unknown_38:$

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:%

unknown_46:?


unknown_47:	?


unknown_48:	?


unknown_49:	?


unknown_50:	?


unknown_51:	?


unknown_52:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./0123456*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_EfficientNet_layer_call_and_return_conditional_losses_43359o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:????????? ?
!
_user_specified_name	input_1
?
p
R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_43034

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:??????????
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-43027*J
_output_shapes8
6:?????????:?????????d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_46128

inputs

identity_1U
SigmoidSigmoidinputs*
T0*0
_output_shapes
:?????????? Z
mulMulinputsSigmoid:y:0*
T0*0
_output_shapes
:?????????? X
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:?????????? ?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-46121*L
_output_shapes:
8:?????????? :?????????? e

Identity_1IdentityIdentityN:output:0*
T0*0
_output_shapes
:?????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42396

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_47796
result_grads_0
result_grads_1
sigmoid_inputs
identityn
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ^
mulMulsigmoid_inputssub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
??
?=
 __inference__wrapped_model_42215
input_1O
5efficientnet_stem_conv_conv2d_readvariableop_resource: :
,efficientnet_stem_bn_readvariableop_resource: <
.efficientnet_stem_bn_readvariableop_1_resource: K
=efficientnet_stem_bn_fusedbatchnormv3_readvariableop_resource: M
?efficientnet_stem_bn_fusedbatchnormv3_readvariableop_1_resource: ^
Defficientnet_stack_0_block0_mb_dw__depthwise_readvariableop_resource: J
<efficientnet_stack_0_block0_mb_dw_bn_readvariableop_resource: L
>efficientnet_stack_0_block0_mb_dw_bn_readvariableop_1_resource: [
Mefficientnet_stack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource: ]
Oefficientnet_stack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource: ^
Defficientnet_stack_0_block0_se_1_conv_conv2d_readvariableop_resource: S
Eefficientnet_stack_0_block0_se_1_conv_biasadd_readvariableop_resource:^
Defficientnet_stack_0_block0_se_2_conv_conv2d_readvariableop_resource: S
Eefficientnet_stack_0_block0_se_2_conv_biasadd_readvariableop_resource: _
Eefficientnet_stack_0_block0_mb_pw_conv_conv2d_readvariableop_resource: J
<efficientnet_stack_0_block0_mb_pw_bn_readvariableop_resource:L
>efficientnet_stack_0_block0_mb_pw_bn_readvariableop_1_resource:[
Mefficientnet_stack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource:]
Oefficientnet_stack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource:^
Defficientnet_stack_1_block0_mb_dw__depthwise_readvariableop_resource:J
<efficientnet_stack_1_block0_mb_dw_bn_readvariableop_resource:L
>efficientnet_stack_1_block0_mb_dw_bn_readvariableop_1_resource:[
Mefficientnet_stack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource:]
Oefficientnet_stack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource:^
Defficientnet_stack_1_block0_se_1_conv_conv2d_readvariableop_resource:S
Eefficientnet_stack_1_block0_se_1_conv_biasadd_readvariableop_resource:^
Defficientnet_stack_1_block0_se_2_conv_conv2d_readvariableop_resource:S
Eefficientnet_stack_1_block0_se_2_conv_biasadd_readvariableop_resource:_
Eefficientnet_stack_1_block0_mb_pw_conv_conv2d_readvariableop_resource:J
<efficientnet_stack_1_block0_mb_pw_bn_readvariableop_resource:L
>efficientnet_stack_1_block0_mb_pw_bn_readvariableop_1_resource:[
Mefficientnet_stack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource:]
Oefficientnet_stack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource:^
Defficientnet_stack_1_block1_mb_dw__depthwise_readvariableop_resource:J
<efficientnet_stack_1_block1_mb_dw_bn_readvariableop_resource:L
>efficientnet_stack_1_block1_mb_dw_bn_readvariableop_1_resource:[
Mefficientnet_stack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_resource:]
Oefficientnet_stack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource:^
Defficientnet_stack_1_block1_se_1_conv_conv2d_readvariableop_resource:S
Eefficientnet_stack_1_block1_se_1_conv_biasadd_readvariableop_resource:^
Defficientnet_stack_1_block1_se_2_conv_conv2d_readvariableop_resource:S
Eefficientnet_stack_1_block1_se_2_conv_biasadd_readvariableop_resource:_
Eefficientnet_stack_1_block1_mb_pw_conv_conv2d_readvariableop_resource:J
<efficientnet_stack_1_block1_mb_pw_bn_readvariableop_resource:L
>efficientnet_stack_1_block1_mb_pw_bn_readvariableop_1_resource:[
Mefficientnet_stack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_resource:]
Oefficientnet_stack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource:P
5efficientnet_post_conv_conv2d_readvariableop_resource:?
;
,efficientnet_post_bn_readvariableop_resource:	?
=
.efficientnet_post_bn_readvariableop_1_resource:	?
L
=efficientnet_post_bn_fusedbatchnormv3_readvariableop_resource:	?
N
?efficientnet_post_bn_fusedbatchnormv3_readvariableop_1_resource:	?
J
7efficientnet_predictions_matmul_readvariableop_resource:	?
F
8efficientnet_predictions_biasadd_readvariableop_resource:
identity??4EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp?6EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp_1?#EfficientNet/post_bn/ReadVariableOp?%EfficientNet/post_bn/ReadVariableOp_1?,EfficientNet/post_conv/Conv2D/ReadVariableOp?/EfficientNet/predictions/BiasAdd/ReadVariableOp?.EfficientNet/predictions/MatMul/ReadVariableOp?;EfficientNet/stack_0_block0_MB_dw_/depthwise/ReadVariableOp?DEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp?FEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1?3EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp?5EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp_1?DEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp?FEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1?3EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp?5EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp_1?<EfficientNet/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp?<EfficientNet/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp?;EfficientNet/stack_0_block0_se_1_conv/Conv2D/ReadVariableOp?<EfficientNet/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp?;EfficientNet/stack_0_block0_se_2_conv/Conv2D/ReadVariableOp?;EfficientNet/stack_1_block0_MB_dw_/depthwise/ReadVariableOp?DEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp?FEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1?3EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp?5EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp_1?DEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp?FEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1?3EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp?5EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp_1?<EfficientNet/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp?<EfficientNet/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp?;EfficientNet/stack_1_block0_se_1_conv/Conv2D/ReadVariableOp?<EfficientNet/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp?;EfficientNet/stack_1_block0_se_2_conv/Conv2D/ReadVariableOp?;EfficientNet/stack_1_block1_MB_dw_/depthwise/ReadVariableOp?DEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp?FEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1?3EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp?5EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp_1?DEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp?FEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1?3EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp?5EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp_1?<EfficientNet/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp?<EfficientNet/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp?;EfficientNet/stack_1_block1_se_1_conv/Conv2D/ReadVariableOp?<EfficientNet/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp?;EfficientNet/stack_1_block1_se_2_conv/Conv2D/ReadVariableOp?4EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp?6EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp_1?#EfficientNet/stem_bn/ReadVariableOp?%EfficientNet/stem_bn/ReadVariableOp_1?,EfficientNet/stem_conv/Conv2D/ReadVariableOp?
,EfficientNet/stem_conv/Conv2D/ReadVariableOpReadVariableOp5efficientnet_stem_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
EfficientNet/stem_conv/Conv2DConv2Dinput_14EfficientNet/stem_conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
#EfficientNet/stem_bn/ReadVariableOpReadVariableOp,efficientnet_stem_bn_readvariableop_resource*
_output_shapes
: *
dtype0?
%EfficientNet/stem_bn/ReadVariableOp_1ReadVariableOp.efficientnet_stem_bn_readvariableop_1_resource*
_output_shapes
: *
dtype0?
4EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp=efficientnet_stem_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
6EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?efficientnet_stem_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
%EfficientNet/stem_bn/FusedBatchNormV3FusedBatchNormV3&EfficientNet/stem_conv/Conv2D:output:0+EfficientNet/stem_bn/ReadVariableOp:value:0-EfficientNet/stem_bn/ReadVariableOp_1:value:0<EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp:value:0>EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
is_training( ?
EfficientNet/stem_swish/SigmoidSigmoid)EfficientNet/stem_bn/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????? ?
EfficientNet/stem_swish/mulMul)EfficientNet/stem_bn/FusedBatchNormV3:y:0#EfficientNet/stem_swish/Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
 EfficientNet/stem_swish/IdentityIdentityEfficientNet/stem_swish/mul:z:0*
T0*0
_output_shapes
:?????????? ?
!EfficientNet/stem_swish/IdentityN	IdentityNEfficientNet/stem_swish/mul:z:0)EfficientNet/stem_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-41981*L
_output_shapes:
8:?????????? :?????????? ?
;EfficientNet/stack_0_block0_MB_dw_/depthwise/ReadVariableOpReadVariableOpDefficientnet_stack_0_block0_mb_dw__depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0?
2EfficientNet/stack_0_block0_MB_dw_/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             ?
:EfficientNet/stack_0_block0_MB_dw_/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
,EfficientNet/stack_0_block0_MB_dw_/depthwiseDepthwiseConv2dNative*EfficientNet/stem_swish/IdentityN:output:0CEfficientNet/stack_0_block0_MB_dw_/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
3EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOpReadVariableOp<efficientnet_stack_0_block0_mb_dw_bn_readvariableop_resource*
_output_shapes
: *
dtype0?
5EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp_1ReadVariableOp>efficientnet_stack_0_block0_mb_dw_bn_readvariableop_1_resource*
_output_shapes
: *
dtype0?
DEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOpMefficientnet_stack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
FEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOefficientnet_stack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5EfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3FusedBatchNormV35EfficientNet/stack_0_block0_MB_dw_/depthwise:output:0;EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp:value:0=EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp_1:value:0LEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:value:0NEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
is_training( ?
/EfficientNet/stack_0_block0_MB_dw_swish/SigmoidSigmoid9EfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????? ?
+EfficientNet/stack_0_block0_MB_dw_swish/mulMul9EfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3:y:03EfficientNet/stack_0_block0_MB_dw_swish/Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
0EfficientNet/stack_0_block0_MB_dw_swish/IdentityIdentity/EfficientNet/stack_0_block0_MB_dw_swish/mul:z:0*
T0*0
_output_shapes
:?????????? ?
1EfficientNet/stack_0_block0_MB_dw_swish/IdentityN	IdentityN/EfficientNet/stack_0_block0_MB_dw_swish/mul:z:09EfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-42006*L
_output_shapes:
8:?????????? :?????????? ?
7EfficientNet/tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
%EfficientNet/tf.math.reduce_mean/MeanMean:EfficientNet/stack_0_block0_MB_dw_swish/IdentityN:output:0@EfficientNet/tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(?
;EfficientNet/stack_0_block0_se_1_conv/Conv2D/ReadVariableOpReadVariableOpDefficientnet_stack_0_block0_se_1_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
,EfficientNet/stack_0_block0_se_1_conv/Conv2DConv2D.EfficientNet/tf.math.reduce_mean/Mean:output:0CEfficientNet/stack_0_block0_se_1_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
<EfficientNet/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOpReadVariableOpEefficientnet_stack_0_block0_se_1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
-EfficientNet/stack_0_block0_se_1_conv/BiasAddBiasAdd5EfficientNet/stack_0_block0_se_1_conv/Conv2D:output:0DEfficientNet/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
,EfficientNet/stack_0_block0_se_swish/SigmoidSigmoid6EfficientNet/stack_0_block0_se_1_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
(EfficientNet/stack_0_block0_se_swish/mulMul6EfficientNet/stack_0_block0_se_1_conv/BiasAdd:output:00EfficientNet/stack_0_block0_se_swish/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
-EfficientNet/stack_0_block0_se_swish/IdentityIdentity,EfficientNet/stack_0_block0_se_swish/mul:z:0*
T0*/
_output_shapes
:??????????
.EfficientNet/stack_0_block0_se_swish/IdentityN	IdentityN,EfficientNet/stack_0_block0_se_swish/mul:z:06EfficientNet/stack_0_block0_se_1_conv/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-42020*J
_output_shapes8
6:?????????:??????????
;EfficientNet/stack_0_block0_se_2_conv/Conv2D/ReadVariableOpReadVariableOpDefficientnet_stack_0_block0_se_2_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
,EfficientNet/stack_0_block0_se_2_conv/Conv2DConv2D7EfficientNet/stack_0_block0_se_swish/IdentityN:output:0CEfficientNet/stack_0_block0_se_2_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
<EfficientNet/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOpReadVariableOpEefficientnet_stack_0_block0_se_2_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
-EfficientNet/stack_0_block0_se_2_conv/BiasAddBiasAdd5EfficientNet/stack_0_block0_se_2_conv/Conv2D:output:0DEfficientNet/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
.EfficientNet/stack_0_block0_se_sigmoid/SigmoidSigmoid6EfficientNet/stack_0_block0_se_2_conv/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
&EfficientNet/stack_0_block0_se_out/mulMul:EfficientNet/stack_0_block0_MB_dw_swish/IdentityN:output:02EfficientNet/stack_0_block0_se_sigmoid/Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
<EfficientNet/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOpReadVariableOpEefficientnet_stack_0_block0_mb_pw_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
-EfficientNet/stack_0_block0_MB_pw_conv/Conv2DConv2D*EfficientNet/stack_0_block0_se_out/mul:z:0DEfficientNet/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
3EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOpReadVariableOp<efficientnet_stack_0_block0_mb_pw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp_1ReadVariableOp>efficientnet_stack_0_block0_mb_pw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
DEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOpMefficientnet_stack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
FEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOefficientnet_stack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3FusedBatchNormV36EfficientNet/stack_0_block0_MB_pw_conv/Conv2D:output:0;EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp:value:0=EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp_1:value:0LEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:value:0NEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( ?
;EfficientNet/stack_1_block0_MB_dw_/depthwise/ReadVariableOpReadVariableOpDefficientnet_stack_1_block0_mb_dw__depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0?
2EfficientNet/stack_1_block0_MB_dw_/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            ?
:EfficientNet/stack_1_block0_MB_dw_/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
,EfficientNet/stack_1_block0_MB_dw_/depthwiseDepthwiseConv2dNative9EfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3:y:0CEfficientNet/stack_1_block0_MB_dw_/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
?
3EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOpReadVariableOp<efficientnet_stack_1_block0_mb_dw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp_1ReadVariableOp>efficientnet_stack_1_block0_mb_dw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
DEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOpMefficientnet_stack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
FEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOefficientnet_stack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3FusedBatchNormV35EfficientNet/stack_1_block0_MB_dw_/depthwise:output:0;EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp:value:0=EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp_1:value:0LEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:value:0NEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( ?
/EfficientNet/stack_1_block0_MB_dw_swish/SigmoidSigmoid9EfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????`?
+EfficientNet/stack_1_block0_MB_dw_swish/mulMul9EfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3:y:03EfficientNet/stack_1_block0_MB_dw_swish/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
0EfficientNet/stack_1_block0_MB_dw_swish/IdentityIdentity/EfficientNet/stack_1_block0_MB_dw_swish/mul:z:0*
T0*/
_output_shapes
:?????????`?
1EfficientNet/stack_1_block0_MB_dw_swish/IdentityN	IdentityN/EfficientNet/stack_1_block0_MB_dw_swish/mul:z:09EfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-42070*J
_output_shapes8
6:?????????`:?????????`?
9EfficientNet/tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
'EfficientNet/tf.math.reduce_mean_1/MeanMean:EfficientNet/stack_1_block0_MB_dw_swish/IdentityN:output:0BEfficientNet/tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
;EfficientNet/stack_1_block0_se_1_conv/Conv2D/ReadVariableOpReadVariableOpDefficientnet_stack_1_block0_se_1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
,EfficientNet/stack_1_block0_se_1_conv/Conv2DConv2D0EfficientNet/tf.math.reduce_mean_1/Mean:output:0CEfficientNet/stack_1_block0_se_1_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
<EfficientNet/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOpReadVariableOpEefficientnet_stack_1_block0_se_1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
-EfficientNet/stack_1_block0_se_1_conv/BiasAddBiasAdd5EfficientNet/stack_1_block0_se_1_conv/Conv2D:output:0DEfficientNet/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
,EfficientNet/stack_1_block0_se_swish/SigmoidSigmoid6EfficientNet/stack_1_block0_se_1_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
(EfficientNet/stack_1_block0_se_swish/mulMul6EfficientNet/stack_1_block0_se_1_conv/BiasAdd:output:00EfficientNet/stack_1_block0_se_swish/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
-EfficientNet/stack_1_block0_se_swish/IdentityIdentity,EfficientNet/stack_1_block0_se_swish/mul:z:0*
T0*/
_output_shapes
:??????????
.EfficientNet/stack_1_block0_se_swish/IdentityN	IdentityN,EfficientNet/stack_1_block0_se_swish/mul:z:06EfficientNet/stack_1_block0_se_1_conv/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-42084*J
_output_shapes8
6:?????????:??????????
;EfficientNet/stack_1_block0_se_2_conv/Conv2D/ReadVariableOpReadVariableOpDefficientnet_stack_1_block0_se_2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
,EfficientNet/stack_1_block0_se_2_conv/Conv2DConv2D7EfficientNet/stack_1_block0_se_swish/IdentityN:output:0CEfficientNet/stack_1_block0_se_2_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
<EfficientNet/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOpReadVariableOpEefficientnet_stack_1_block0_se_2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
-EfficientNet/stack_1_block0_se_2_conv/BiasAddBiasAdd5EfficientNet/stack_1_block0_se_2_conv/Conv2D:output:0DEfficientNet/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
.EfficientNet/stack_1_block0_se_sigmoid/SigmoidSigmoid6EfficientNet/stack_1_block0_se_2_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
&EfficientNet/stack_1_block0_se_out/mulMul:EfficientNet/stack_1_block0_MB_dw_swish/IdentityN:output:02EfficientNet/stack_1_block0_se_sigmoid/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
<EfficientNet/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOpReadVariableOpEefficientnet_stack_1_block0_mb_pw_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
-EfficientNet/stack_1_block0_MB_pw_conv/Conv2DConv2D*EfficientNet/stack_1_block0_se_out/mul:z:0DEfficientNet/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
?
3EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOpReadVariableOp<efficientnet_stack_1_block0_mb_pw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp_1ReadVariableOp>efficientnet_stack_1_block0_mb_pw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
DEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOpMefficientnet_stack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
FEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOefficientnet_stack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3FusedBatchNormV36EfficientNet/stack_1_block0_MB_pw_conv/Conv2D:output:0;EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp:value:0=EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp_1:value:0LEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:value:0NEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( ?
;EfficientNet/stack_1_block1_MB_dw_/depthwise/ReadVariableOpReadVariableOpDefficientnet_stack_1_block1_mb_dw__depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0?
2EfficientNet/stack_1_block1_MB_dw_/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            ?
:EfficientNet/stack_1_block1_MB_dw_/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
,EfficientNet/stack_1_block1_MB_dw_/depthwiseDepthwiseConv2dNative9EfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3:y:0CEfficientNet/stack_1_block1_MB_dw_/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
?
3EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOpReadVariableOp<efficientnet_stack_1_block1_mb_dw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp_1ReadVariableOp>efficientnet_stack_1_block1_mb_dw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
DEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOpMefficientnet_stack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
FEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOefficientnet_stack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3FusedBatchNormV35EfficientNet/stack_1_block1_MB_dw_/depthwise:output:0;EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp:value:0=EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp_1:value:0LEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:value:0NEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( ?
/EfficientNet/stack_1_block1_MB_dw_swish/SigmoidSigmoid9EfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????`?
+EfficientNet/stack_1_block1_MB_dw_swish/mulMul9EfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3:y:03EfficientNet/stack_1_block1_MB_dw_swish/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
0EfficientNet/stack_1_block1_MB_dw_swish/IdentityIdentity/EfficientNet/stack_1_block1_MB_dw_swish/mul:z:0*
T0*/
_output_shapes
:?????????`?
1EfficientNet/stack_1_block1_MB_dw_swish/IdentityN	IdentityN/EfficientNet/stack_1_block1_MB_dw_swish/mul:z:09EfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-42134*J
_output_shapes8
6:?????????`:?????????`?
9EfficientNet/tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
'EfficientNet/tf.math.reduce_mean_2/MeanMean:EfficientNet/stack_1_block1_MB_dw_swish/IdentityN:output:0BEfficientNet/tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
;EfficientNet/stack_1_block1_se_1_conv/Conv2D/ReadVariableOpReadVariableOpDefficientnet_stack_1_block1_se_1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
,EfficientNet/stack_1_block1_se_1_conv/Conv2DConv2D0EfficientNet/tf.math.reduce_mean_2/Mean:output:0CEfficientNet/stack_1_block1_se_1_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
<EfficientNet/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOpReadVariableOpEefficientnet_stack_1_block1_se_1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
-EfficientNet/stack_1_block1_se_1_conv/BiasAddBiasAdd5EfficientNet/stack_1_block1_se_1_conv/Conv2D:output:0DEfficientNet/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
,EfficientNet/stack_1_block1_se_swish/SigmoidSigmoid6EfficientNet/stack_1_block1_se_1_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
(EfficientNet/stack_1_block1_se_swish/mulMul6EfficientNet/stack_1_block1_se_1_conv/BiasAdd:output:00EfficientNet/stack_1_block1_se_swish/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
-EfficientNet/stack_1_block1_se_swish/IdentityIdentity,EfficientNet/stack_1_block1_se_swish/mul:z:0*
T0*/
_output_shapes
:??????????
.EfficientNet/stack_1_block1_se_swish/IdentityN	IdentityN,EfficientNet/stack_1_block1_se_swish/mul:z:06EfficientNet/stack_1_block1_se_1_conv/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-42148*J
_output_shapes8
6:?????????:??????????
;EfficientNet/stack_1_block1_se_2_conv/Conv2D/ReadVariableOpReadVariableOpDefficientnet_stack_1_block1_se_2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
,EfficientNet/stack_1_block1_se_2_conv/Conv2DConv2D7EfficientNet/stack_1_block1_se_swish/IdentityN:output:0CEfficientNet/stack_1_block1_se_2_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
<EfficientNet/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOpReadVariableOpEefficientnet_stack_1_block1_se_2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
-EfficientNet/stack_1_block1_se_2_conv/BiasAddBiasAdd5EfficientNet/stack_1_block1_se_2_conv/Conv2D:output:0DEfficientNet/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
.EfficientNet/stack_1_block1_se_sigmoid/SigmoidSigmoid6EfficientNet/stack_1_block1_se_2_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
&EfficientNet/stack_1_block1_se_out/mulMul:EfficientNet/stack_1_block1_MB_dw_swish/IdentityN:output:02EfficientNet/stack_1_block1_se_sigmoid/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
<EfficientNet/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOpReadVariableOpEefficientnet_stack_1_block1_mb_pw_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
-EfficientNet/stack_1_block1_MB_pw_conv/Conv2DConv2D*EfficientNet/stack_1_block1_se_out/mul:z:0DEfficientNet/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
?
3EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOpReadVariableOp<efficientnet_stack_1_block1_mb_pw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp_1ReadVariableOp>efficientnet_stack_1_block1_mb_pw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
DEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOpMefficientnet_stack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
FEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOefficientnet_stack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5EfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3FusedBatchNormV36EfficientNet/stack_1_block1_MB_pw_conv/Conv2D:output:0;EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp:value:0=EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp_1:value:0LEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:value:0NEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( ?
-EfficientNet/stack_1_block1_dropdrop/IdentityIdentity9EfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????`?
&EfficientNet/stack_1_block1_output/addAddV29EfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3:y:06EfficientNet/stack_1_block1_dropdrop/Identity:output:0*
T0*/
_output_shapes
:?????????`?
,EfficientNet/post_conv/Conv2D/ReadVariableOpReadVariableOp5efficientnet_post_conv_conv2d_readvariableop_resource*'
_output_shapes
:?
*
dtype0?
EfficientNet/post_conv/Conv2DConv2D*EfficientNet/stack_1_block1_output/add:z:04EfficientNet/post_conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`?
*
paddingVALID*
strides
?
#EfficientNet/post_bn/ReadVariableOpReadVariableOp,efficientnet_post_bn_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
%EfficientNet/post_bn/ReadVariableOp_1ReadVariableOp.efficientnet_post_bn_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
4EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp=efficientnet_post_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
6EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp?efficientnet_post_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
%EfficientNet/post_bn/FusedBatchNormV3FusedBatchNormV3&EfficientNet/post_conv/Conv2D:output:0+EfficientNet/post_bn/ReadVariableOp:value:0-EfficientNet/post_bn/ReadVariableOp_1:value:0<EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp:value:0>EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????`?
:?
:?
:?
:?
:*
epsilon%o?:*
is_training( ?
EfficientNet/post_swish/SigmoidSigmoid)EfficientNet/post_bn/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`?
?
EfficientNet/post_swish/mulMul)EfficientNet/post_bn/FusedBatchNormV3:y:0#EfficientNet/post_swish/Sigmoid:y:0*
T0*0
_output_shapes
:?????????`?
?
 EfficientNet/post_swish/IdentityIdentityEfficientNet/post_swish/mul:z:0*
T0*0
_output_shapes
:?????????`?
?
!EfficientNet/post_swish/IdentityN	IdentityNEfficientNet/post_swish/mul:z:0)EfficientNet/post_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-42198*L
_output_shapes:
8:?????????`?
:?????????`?
}
,EfficientNet/avg_pool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
EfficientNet/avg_pool/MeanMean*EfficientNet/post_swish/IdentityN:output:05EfficientNet/avg_pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????
?
EfficientNet/head_drop/IdentityIdentity#EfficientNet/avg_pool/Mean:output:0*
T0*(
_output_shapes
:??????????
?
.EfficientNet/predictions/MatMul/ReadVariableOpReadVariableOp7efficientnet_predictions_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
EfficientNet/predictions/MatMulMatMul(EfficientNet/head_drop/Identity:output:06EfficientNet/predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
/EfficientNet/predictions/BiasAdd/ReadVariableOpReadVariableOp8efficientnet_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 EfficientNet/predictions/BiasAddBiasAdd)EfficientNet/predictions/MatMul:product:07EfficientNet/predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 EfficientNet/predictions/SoftmaxSoftmax)EfficientNet/predictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????y
IdentityIdentity*EfficientNet/predictions/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp5^EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp7^EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp_1$^EfficientNet/post_bn/ReadVariableOp&^EfficientNet/post_bn/ReadVariableOp_1-^EfficientNet/post_conv/Conv2D/ReadVariableOp0^EfficientNet/predictions/BiasAdd/ReadVariableOp/^EfficientNet/predictions/MatMul/ReadVariableOp<^EfficientNet/stack_0_block0_MB_dw_/depthwise/ReadVariableOpE^EfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpG^EfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_14^EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp6^EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp_1E^EfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpG^EfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_14^EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp6^EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp_1=^EfficientNet/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp=^EfficientNet/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp<^EfficientNet/stack_0_block0_se_1_conv/Conv2D/ReadVariableOp=^EfficientNet/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp<^EfficientNet/stack_0_block0_se_2_conv/Conv2D/ReadVariableOp<^EfficientNet/stack_1_block0_MB_dw_/depthwise/ReadVariableOpE^EfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpG^EfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_14^EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp6^EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp_1E^EfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpG^EfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_14^EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp6^EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp_1=^EfficientNet/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp=^EfficientNet/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp<^EfficientNet/stack_1_block0_se_1_conv/Conv2D/ReadVariableOp=^EfficientNet/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp<^EfficientNet/stack_1_block0_se_2_conv/Conv2D/ReadVariableOp<^EfficientNet/stack_1_block1_MB_dw_/depthwise/ReadVariableOpE^EfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOpG^EfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_14^EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp6^EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp_1E^EfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOpG^EfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_14^EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp6^EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp_1=^EfficientNet/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp=^EfficientNet/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp<^EfficientNet/stack_1_block1_se_1_conv/Conv2D/ReadVariableOp=^EfficientNet/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp<^EfficientNet/stack_1_block1_se_2_conv/Conv2D/ReadVariableOp5^EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp7^EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp_1$^EfficientNet/stem_bn/ReadVariableOp&^EfficientNet/stem_bn/ReadVariableOp_1-^EfficientNet/stem_conv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2l
4EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp4EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp2p
6EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp_16EfficientNet/post_bn/FusedBatchNormV3/ReadVariableOp_12J
#EfficientNet/post_bn/ReadVariableOp#EfficientNet/post_bn/ReadVariableOp2N
%EfficientNet/post_bn/ReadVariableOp_1%EfficientNet/post_bn/ReadVariableOp_12\
,EfficientNet/post_conv/Conv2D/ReadVariableOp,EfficientNet/post_conv/Conv2D/ReadVariableOp2b
/EfficientNet/predictions/BiasAdd/ReadVariableOp/EfficientNet/predictions/BiasAdd/ReadVariableOp2`
.EfficientNet/predictions/MatMul/ReadVariableOp.EfficientNet/predictions/MatMul/ReadVariableOp2z
;EfficientNet/stack_0_block0_MB_dw_/depthwise/ReadVariableOp;EfficientNet/stack_0_block0_MB_dw_/depthwise/ReadVariableOp2?
DEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpDEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp2?
FEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1FEfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_12j
3EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp3EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp2n
5EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp_15EfficientNet/stack_0_block0_MB_dw_bn/ReadVariableOp_12?
DEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpDEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp2?
FEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1FEfficientNet/stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_12j
3EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp3EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp2n
5EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp_15EfficientNet/stack_0_block0_MB_pw_bn/ReadVariableOp_12|
<EfficientNet/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp<EfficientNet/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp2|
<EfficientNet/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp<EfficientNet/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp2z
;EfficientNet/stack_0_block0_se_1_conv/Conv2D/ReadVariableOp;EfficientNet/stack_0_block0_se_1_conv/Conv2D/ReadVariableOp2|
<EfficientNet/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp<EfficientNet/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp2z
;EfficientNet/stack_0_block0_se_2_conv/Conv2D/ReadVariableOp;EfficientNet/stack_0_block0_se_2_conv/Conv2D/ReadVariableOp2z
;EfficientNet/stack_1_block0_MB_dw_/depthwise/ReadVariableOp;EfficientNet/stack_1_block0_MB_dw_/depthwise/ReadVariableOp2?
DEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpDEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp2?
FEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1FEfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_12j
3EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp3EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp2n
5EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp_15EfficientNet/stack_1_block0_MB_dw_bn/ReadVariableOp_12?
DEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpDEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp2?
FEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1FEfficientNet/stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_12j
3EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp3EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp2n
5EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp_15EfficientNet/stack_1_block0_MB_pw_bn/ReadVariableOp_12|
<EfficientNet/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp<EfficientNet/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp2|
<EfficientNet/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp<EfficientNet/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp2z
;EfficientNet/stack_1_block0_se_1_conv/Conv2D/ReadVariableOp;EfficientNet/stack_1_block0_se_1_conv/Conv2D/ReadVariableOp2|
<EfficientNet/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp<EfficientNet/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp2z
;EfficientNet/stack_1_block0_se_2_conv/Conv2D/ReadVariableOp;EfficientNet/stack_1_block0_se_2_conv/Conv2D/ReadVariableOp2z
;EfficientNet/stack_1_block1_MB_dw_/depthwise/ReadVariableOp;EfficientNet/stack_1_block1_MB_dw_/depthwise/ReadVariableOp2?
DEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOpDEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp2?
FEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1FEfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_12j
3EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp3EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp2n
5EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp_15EfficientNet/stack_1_block1_MB_dw_bn/ReadVariableOp_12?
DEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOpDEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp2?
FEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1FEfficientNet/stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_12j
3EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp3EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp2n
5EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp_15EfficientNet/stack_1_block1_MB_pw_bn/ReadVariableOp_12|
<EfficientNet/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp<EfficientNet/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp2|
<EfficientNet/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp<EfficientNet/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp2z
;EfficientNet/stack_1_block1_se_1_conv/Conv2D/ReadVariableOp;EfficientNet/stack_1_block1_se_1_conv/Conv2D/ReadVariableOp2|
<EfficientNet/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp<EfficientNet/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp2z
;EfficientNet/stack_1_block1_se_2_conv/Conv2D/ReadVariableOp;EfficientNet/stack_1_block1_se_2_conv/Conv2D/ReadVariableOp2l
4EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp4EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp2p
6EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp_16EfficientNet/stem_bn/FusedBatchNormV3/ReadVariableOp_12J
#EfficientNet/stem_bn/ReadVariableOp#EfficientNet/stem_bn/ReadVariableOp2N
%EfficientNet/stem_bn/ReadVariableOp_1%EfficientNet/stem_bn/ReadVariableOp_12\
,EfficientNet/stem_conv/Conv2D/ReadVariableOp,EfficientNet/stem_conv/Conv2D/ReadVariableOp:Y U
0
_output_shapes
:????????? ?
!
_user_specified_name	input_1
?	
?
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46002

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42301?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46287

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
p
R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_46539

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:??????????
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-46532*J
_output_shapes8
6:?????????:?????????d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42429

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47736
result_grads_0
result_grads_1A
=sigmoid_efficientnet_stack_1_block1_mb_dw_bn_fusedbatchnormv3
identity?
SigmoidSigmoid=sigmoid_efficientnet_stack_1_block1_mb_dw_bn_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
mulMul=sigmoid_efficientnet_stack_1_block1_mb_dw_bn_fusedbatchnormv3sub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?
?
"__inference_internal_grad_fn_47751
result_grads_0
result_grads_19
5sigmoid_efficientnet_stack_1_block1_se_1_conv_biasadd
identity?
SigmoidSigmoid5sigmoid_efficientnet_stack_1_block1_se_1_conv_biasadd^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:??????????
mulMul5sigmoid_efficientnet_stack_1_block1_se_1_conv_biasaddsub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?

?
"__inference_internal_grad_fn_48021
result_grads_0
result_grads_1$
 sigmoid_stem_bn_fusedbatchnormv3
identity?
SigmoidSigmoid sigmoid_stem_bn_fusedbatchnormv3^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? p
mulMul sigmoid_stem_bn_fusedbatchnormv3sub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
?
?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46341

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46620

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42524?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
p
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_43267

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????`c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
8__inference_stack_1_block0_se_1_conv_layer_call_fn_46514

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block0_se_1_conv_layer_call_and_return_conditional_losses_43018w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
P__inference_stack_0_block0_MB_dw__layer_call_and_return_conditional_losses_42806

inputs;
!depthwise_readvariableop_resource: 
identity??depthwise/ReadVariableOp?
depthwise/ReadVariableOpReadVariableOp!depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0h
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             h
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
j
IdentityIdentitydepthwise:output:0^NoOp*
T0*0
_output_shapes
:?????????? a
NoOpNoOp^depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????? : 24
depthwise/ReadVariableOpdepthwise/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?

?
S__inference_stack_0_block0_se_2_conv_layer_call_and_return_conditional_losses_42889

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46405

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42984w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
'__inference_stem_bn_layer_call_fn_45873

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_stem_bn_layer_call_and_return_conditional_losses_42775x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
??
?2
G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544

inputsB
(stem_conv_conv2d_readvariableop_resource: -
stem_bn_readvariableop_resource: /
!stem_bn_readvariableop_1_resource: >
0stem_bn_fusedbatchnormv3_readvariableop_resource: @
2stem_bn_fusedbatchnormv3_readvariableop_1_resource: Q
7stack_0_block0_mb_dw__depthwise_readvariableop_resource: =
/stack_0_block0_mb_dw_bn_readvariableop_resource: ?
1stack_0_block0_mb_dw_bn_readvariableop_1_resource: N
@stack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource: P
Bstack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource: Q
7stack_0_block0_se_1_conv_conv2d_readvariableop_resource: F
8stack_0_block0_se_1_conv_biasadd_readvariableop_resource:Q
7stack_0_block0_se_2_conv_conv2d_readvariableop_resource: F
8stack_0_block0_se_2_conv_biasadd_readvariableop_resource: R
8stack_0_block0_mb_pw_conv_conv2d_readvariableop_resource: =
/stack_0_block0_mb_pw_bn_readvariableop_resource:?
1stack_0_block0_mb_pw_bn_readvariableop_1_resource:N
@stack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource:Q
7stack_1_block0_mb_dw__depthwise_readvariableop_resource:=
/stack_1_block0_mb_dw_bn_readvariableop_resource:?
1stack_1_block0_mb_dw_bn_readvariableop_1_resource:N
@stack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource:Q
7stack_1_block0_se_1_conv_conv2d_readvariableop_resource:F
8stack_1_block0_se_1_conv_biasadd_readvariableop_resource:Q
7stack_1_block0_se_2_conv_conv2d_readvariableop_resource:F
8stack_1_block0_se_2_conv_biasadd_readvariableop_resource:R
8stack_1_block0_mb_pw_conv_conv2d_readvariableop_resource:=
/stack_1_block0_mb_pw_bn_readvariableop_resource:?
1stack_1_block0_mb_pw_bn_readvariableop_1_resource:N
@stack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource:Q
7stack_1_block1_mb_dw__depthwise_readvariableop_resource:=
/stack_1_block1_mb_dw_bn_readvariableop_resource:?
1stack_1_block1_mb_dw_bn_readvariableop_1_resource:N
@stack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource:Q
7stack_1_block1_se_1_conv_conv2d_readvariableop_resource:F
8stack_1_block1_se_1_conv_biasadd_readvariableop_resource:Q
7stack_1_block1_se_2_conv_conv2d_readvariableop_resource:F
8stack_1_block1_se_2_conv_biasadd_readvariableop_resource:R
8stack_1_block1_mb_pw_conv_conv2d_readvariableop_resource:=
/stack_1_block1_mb_pw_bn_readvariableop_resource:?
1stack_1_block1_mb_pw_bn_readvariableop_1_resource:N
@stack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_resource:P
Bstack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource:C
(post_conv_conv2d_readvariableop_resource:?
.
post_bn_readvariableop_resource:	?
0
!post_bn_readvariableop_1_resource:	?
?
0post_bn_fusedbatchnormv3_readvariableop_resource:	?
A
2post_bn_fusedbatchnormv3_readvariableop_1_resource:	?
=
*predictions_matmul_readvariableop_resource:	?
9
+predictions_biasadd_readvariableop_resource:
identity??'post_bn/FusedBatchNormV3/ReadVariableOp?)post_bn/FusedBatchNormV3/ReadVariableOp_1?post_bn/ReadVariableOp?post_bn/ReadVariableOp_1?post_conv/Conv2D/ReadVariableOp?"predictions/BiasAdd/ReadVariableOp?!predictions/MatMul/ReadVariableOp?.stack_0_block0_MB_dw_/depthwise/ReadVariableOp?7stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp?9stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_0_block0_MB_dw_bn/ReadVariableOp?(stack_0_block0_MB_dw_bn/ReadVariableOp_1?7stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp?9stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_0_block0_MB_pw_bn/ReadVariableOp?(stack_0_block0_MB_pw_bn/ReadVariableOp_1?/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp?/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp?.stack_0_block0_se_1_conv/Conv2D/ReadVariableOp?/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp?.stack_0_block0_se_2_conv/Conv2D/ReadVariableOp?.stack_1_block0_MB_dw_/depthwise/ReadVariableOp?7stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp?9stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_1_block0_MB_dw_bn/ReadVariableOp?(stack_1_block0_MB_dw_bn/ReadVariableOp_1?7stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp?9stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_1_block0_MB_pw_bn/ReadVariableOp?(stack_1_block0_MB_pw_bn/ReadVariableOp_1?/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp?/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp?.stack_1_block0_se_1_conv/Conv2D/ReadVariableOp?/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp?.stack_1_block0_se_2_conv/Conv2D/ReadVariableOp?.stack_1_block1_MB_dw_/depthwise/ReadVariableOp?7stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp?9stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_1_block1_MB_dw_bn/ReadVariableOp?(stack_1_block1_MB_dw_bn/ReadVariableOp_1?7stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp?9stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1?&stack_1_block1_MB_pw_bn/ReadVariableOp?(stack_1_block1_MB_pw_bn/ReadVariableOp_1?/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp?/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp?.stack_1_block1_se_1_conv/Conv2D/ReadVariableOp?/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp?.stack_1_block1_se_2_conv/Conv2D/ReadVariableOp?'stem_bn/FusedBatchNormV3/ReadVariableOp?)stem_bn/FusedBatchNormV3/ReadVariableOp_1?stem_bn/ReadVariableOp?stem_bn/ReadVariableOp_1?stem_conv/Conv2D/ReadVariableOp?
stem_conv/Conv2D/ReadVariableOpReadVariableOp(stem_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
stem_conv/Conv2DConv2Dinputs'stem_conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
r
stem_bn/ReadVariableOpReadVariableOpstem_bn_readvariableop_resource*
_output_shapes
: *
dtype0v
stem_bn/ReadVariableOp_1ReadVariableOp!stem_bn_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'stem_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp0stem_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
)stem_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp2stem_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
stem_bn/FusedBatchNormV3FusedBatchNormV3stem_conv/Conv2D:output:0stem_bn/ReadVariableOp:value:0 stem_bn/ReadVariableOp_1:value:0/stem_bn/FusedBatchNormV3/ReadVariableOp:value:01stem_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
is_training( v
stem_swish/SigmoidSigmoidstem_bn/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????? ?
stem_swish/mulMulstem_bn/FusedBatchNormV3:y:0stem_swish/Sigmoid:y:0*
T0*0
_output_shapes
:?????????? n
stem_swish/IdentityIdentitystem_swish/mul:z:0*
T0*0
_output_shapes
:?????????? ?
stem_swish/IdentityN	IdentityNstem_swish/mul:z:0stem_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45310*L
_output_shapes:
8:?????????? :?????????? ?
.stack_0_block0_MB_dw_/depthwise/ReadVariableOpReadVariableOp7stack_0_block0_mb_dw__depthwise_readvariableop_resource*&
_output_shapes
: *
dtype0~
%stack_0_block0_MB_dw_/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             ~
-stack_0_block0_MB_dw_/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
stack_0_block0_MB_dw_/depthwiseDepthwiseConv2dNativestem_swish/IdentityN:output:06stack_0_block0_MB_dw_/depthwise/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingSAME*
strides
?
&stack_0_block0_MB_dw_bn/ReadVariableOpReadVariableOp/stack_0_block0_mb_dw_bn_readvariableop_resource*
_output_shapes
: *
dtype0?
(stack_0_block0_MB_dw_bn/ReadVariableOp_1ReadVariableOp1stack_0_block0_mb_dw_bn_readvariableop_1_resource*
_output_shapes
: *
dtype0?
7stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
9stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_0_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
(stack_0_block0_MB_dw_bn/FusedBatchNormV3FusedBatchNormV3(stack_0_block0_MB_dw_/depthwise:output:0.stack_0_block0_MB_dw_bn/ReadVariableOp:value:00stack_0_block0_MB_dw_bn/ReadVariableOp_1:value:0?stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
is_training( ?
"stack_0_block0_MB_dw_swish/SigmoidSigmoid,stack_0_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????? ?
stack_0_block0_MB_dw_swish/mulMul,stack_0_block0_MB_dw_bn/FusedBatchNormV3:y:0&stack_0_block0_MB_dw_swish/Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
#stack_0_block0_MB_dw_swish/IdentityIdentity"stack_0_block0_MB_dw_swish/mul:z:0*
T0*0
_output_shapes
:?????????? ?
$stack_0_block0_MB_dw_swish/IdentityN	IdentityN"stack_0_block0_MB_dw_swish/mul:z:0,stack_0_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45335*L
_output_shapes:
8:?????????? :?????????? {
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean/MeanMean-stack_0_block0_MB_dw_swish/IdentityN:output:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:????????? *
	keep_dims(?
.stack_0_block0_se_1_conv/Conv2D/ReadVariableOpReadVariableOp7stack_0_block0_se_1_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
stack_0_block0_se_1_conv/Conv2DConv2D!tf.math.reduce_mean/Mean:output:06stack_0_block0_se_1_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_0_block0_se_1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_0_block0_se_1_conv/BiasAddBiasAdd(stack_0_block0_se_1_conv/Conv2D:output:07stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
stack_0_block0_se_swish/SigmoidSigmoid)stack_0_block0_se_1_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_0_block0_se_swish/mulMul)stack_0_block0_se_1_conv/BiasAdd:output:0#stack_0_block0_se_swish/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
 stack_0_block0_se_swish/IdentityIdentitystack_0_block0_se_swish/mul:z:0*
T0*/
_output_shapes
:??????????
!stack_0_block0_se_swish/IdentityN	IdentityNstack_0_block0_se_swish/mul:z:0)stack_0_block0_se_1_conv/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-45349*J
_output_shapes8
6:?????????:??????????
.stack_0_block0_se_2_conv/Conv2D/ReadVariableOpReadVariableOp7stack_0_block0_se_2_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
stack_0_block0_se_2_conv/Conv2DConv2D*stack_0_block0_se_swish/IdentityN:output:06stack_0_block0_se_2_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_0_block0_se_2_conv_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
 stack_0_block0_se_2_conv/BiasAddBiasAdd(stack_0_block0_se_2_conv/Conv2D:output:07stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
!stack_0_block0_se_sigmoid/SigmoidSigmoid)stack_0_block0_se_2_conv/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
stack_0_block0_se_out/mulMul-stack_0_block0_MB_dw_swish/IdentityN:output:0%stack_0_block0_se_sigmoid/Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOpReadVariableOp8stack_0_block0_mb_pw_conv_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
 stack_0_block0_MB_pw_conv/Conv2DConv2Dstack_0_block0_se_out/mul:z:07stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
&stack_0_block0_MB_pw_bn/ReadVariableOpReadVariableOp/stack_0_block0_mb_pw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_0_block0_MB_pw_bn/ReadVariableOp_1ReadVariableOp1stack_0_block0_mb_pw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_0_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_0_block0_MB_pw_bn/FusedBatchNormV3FusedBatchNormV3)stack_0_block0_MB_pw_conv/Conv2D:output:0.stack_0_block0_MB_pw_bn/ReadVariableOp:value:00stack_0_block0_MB_pw_bn/ReadVariableOp_1:value:0?stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:??????????:::::*
epsilon%o?:*
is_training( ?
.stack_1_block0_MB_dw_/depthwise/ReadVariableOpReadVariableOp7stack_1_block0_mb_dw__depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0~
%stack_1_block0_MB_dw_/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            ~
-stack_1_block0_MB_dw_/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
stack_1_block0_MB_dw_/depthwiseDepthwiseConv2dNative,stack_0_block0_MB_pw_bn/FusedBatchNormV3:y:06stack_1_block0_MB_dw_/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
?
&stack_1_block0_MB_dw_bn/ReadVariableOpReadVariableOp/stack_1_block0_mb_dw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_1_block0_MB_dw_bn/ReadVariableOp_1ReadVariableOp1stack_1_block0_mb_dw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_1_block0_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_1_block0_MB_dw_bn/FusedBatchNormV3FusedBatchNormV3(stack_1_block0_MB_dw_/depthwise:output:0.stack_1_block0_MB_dw_bn/ReadVariableOp:value:00stack_1_block0_MB_dw_bn/ReadVariableOp_1:value:0?stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( ?
"stack_1_block0_MB_dw_swish/SigmoidSigmoid,stack_1_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????`?
stack_1_block0_MB_dw_swish/mulMul,stack_1_block0_MB_dw_bn/FusedBatchNormV3:y:0&stack_1_block0_MB_dw_swish/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
#stack_1_block0_MB_dw_swish/IdentityIdentity"stack_1_block0_MB_dw_swish/mul:z:0*
T0*/
_output_shapes
:?????????`?
$stack_1_block0_MB_dw_swish/IdentityN	IdentityN"stack_1_block0_MB_dw_swish/mul:z:0,stack_1_block0_MB_dw_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45399*J
_output_shapes8
6:?????????`:?????????`}
,tf.math.reduce_mean_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_1/MeanMean-stack_1_block0_MB_dw_swish/IdentityN:output:05tf.math.reduce_mean_1/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
.stack_1_block0_se_1_conv/Conv2D/ReadVariableOpReadVariableOp7stack_1_block0_se_1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
stack_1_block0_se_1_conv/Conv2DConv2D#tf.math.reduce_mean_1/Mean:output:06stack_1_block0_se_1_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_1_block0_se_1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_1_block0_se_1_conv/BiasAddBiasAdd(stack_1_block0_se_1_conv/Conv2D:output:07stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
stack_1_block0_se_swish/SigmoidSigmoid)stack_1_block0_se_1_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_1_block0_se_swish/mulMul)stack_1_block0_se_1_conv/BiasAdd:output:0#stack_1_block0_se_swish/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
 stack_1_block0_se_swish/IdentityIdentitystack_1_block0_se_swish/mul:z:0*
T0*/
_output_shapes
:??????????
!stack_1_block0_se_swish/IdentityN	IdentityNstack_1_block0_se_swish/mul:z:0)stack_1_block0_se_1_conv/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-45413*J
_output_shapes8
6:?????????:??????????
.stack_1_block0_se_2_conv/Conv2D/ReadVariableOpReadVariableOp7stack_1_block0_se_2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
stack_1_block0_se_2_conv/Conv2DConv2D*stack_1_block0_se_swish/IdentityN:output:06stack_1_block0_se_2_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_1_block0_se_2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_1_block0_se_2_conv/BiasAddBiasAdd(stack_1_block0_se_2_conv/Conv2D:output:07stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
!stack_1_block0_se_sigmoid/SigmoidSigmoid)stack_1_block0_se_2_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_1_block0_se_out/mulMul-stack_1_block0_MB_dw_swish/IdentityN:output:0%stack_1_block0_se_sigmoid/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOpReadVariableOp8stack_1_block0_mb_pw_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
 stack_1_block0_MB_pw_conv/Conv2DConv2Dstack_1_block0_se_out/mul:z:07stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
?
&stack_1_block0_MB_pw_bn/ReadVariableOpReadVariableOp/stack_1_block0_mb_pw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_1_block0_MB_pw_bn/ReadVariableOp_1ReadVariableOp1stack_1_block0_mb_pw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_1_block0_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_1_block0_MB_pw_bn/FusedBatchNormV3FusedBatchNormV3)stack_1_block0_MB_pw_conv/Conv2D:output:0.stack_1_block0_MB_pw_bn/ReadVariableOp:value:00stack_1_block0_MB_pw_bn/ReadVariableOp_1:value:0?stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( ?
.stack_1_block1_MB_dw_/depthwise/ReadVariableOpReadVariableOp7stack_1_block1_mb_dw__depthwise_readvariableop_resource*&
_output_shapes
:*
dtype0~
%stack_1_block1_MB_dw_/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            ~
-stack_1_block1_MB_dw_/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ?
stack_1_block1_MB_dw_/depthwiseDepthwiseConv2dNative,stack_1_block0_MB_pw_bn/FusedBatchNormV3:y:06stack_1_block1_MB_dw_/depthwise/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingSAME*
strides
?
&stack_1_block1_MB_dw_bn/ReadVariableOpReadVariableOp/stack_1_block1_mb_dw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_1_block1_MB_dw_bn/ReadVariableOp_1ReadVariableOp1stack_1_block1_mb_dw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_1_block1_mb_dw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_1_block1_MB_dw_bn/FusedBatchNormV3FusedBatchNormV3(stack_1_block1_MB_dw_/depthwise:output:0.stack_1_block1_MB_dw_bn/ReadVariableOp:value:00stack_1_block1_MB_dw_bn/ReadVariableOp_1:value:0?stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( ?
"stack_1_block1_MB_dw_swish/SigmoidSigmoid,stack_1_block1_MB_dw_bn/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????`?
stack_1_block1_MB_dw_swish/mulMul,stack_1_block1_MB_dw_bn/FusedBatchNormV3:y:0&stack_1_block1_MB_dw_swish/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
#stack_1_block1_MB_dw_swish/IdentityIdentity"stack_1_block1_MB_dw_swish/mul:z:0*
T0*/
_output_shapes
:?????????`?
$stack_1_block1_MB_dw_swish/IdentityN	IdentityN"stack_1_block1_MB_dw_swish/mul:z:0,stack_1_block1_MB_dw_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45463*J
_output_shapes8
6:?????????`:?????????`}
,tf.math.reduce_mean_2/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
tf.math.reduce_mean_2/MeanMean-stack_1_block1_MB_dw_swish/IdentityN:output:05tf.math.reduce_mean_2/Mean/reduction_indices:output:0*
T0*/
_output_shapes
:?????????*
	keep_dims(?
.stack_1_block1_se_1_conv/Conv2D/ReadVariableOpReadVariableOp7stack_1_block1_se_1_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
stack_1_block1_se_1_conv/Conv2DConv2D#tf.math.reduce_mean_2/Mean:output:06stack_1_block1_se_1_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_1_block1_se_1_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_1_block1_se_1_conv/BiasAddBiasAdd(stack_1_block1_se_1_conv/Conv2D:output:07stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
stack_1_block1_se_swish/SigmoidSigmoid)stack_1_block1_se_1_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_1_block1_se_swish/mulMul)stack_1_block1_se_1_conv/BiasAdd:output:0#stack_1_block1_se_swish/Sigmoid:y:0*
T0*/
_output_shapes
:??????????
 stack_1_block1_se_swish/IdentityIdentitystack_1_block1_se_swish/mul:z:0*
T0*/
_output_shapes
:??????????
!stack_1_block1_se_swish/IdentityN	IdentityNstack_1_block1_se_swish/mul:z:0)stack_1_block1_se_1_conv/BiasAdd:output:0*
T
2*+
_gradient_op_typeCustomGradient-45477*J
_output_shapes8
6:?????????:??????????
.stack_1_block1_se_2_conv/Conv2D/ReadVariableOpReadVariableOp7stack_1_block1_se_2_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
stack_1_block1_se_2_conv/Conv2DConv2D*stack_1_block1_se_swish/IdentityN:output:06stack_1_block1_se_2_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOpReadVariableOp8stack_1_block1_se_2_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
 stack_1_block1_se_2_conv/BiasAddBiasAdd(stack_1_block1_se_2_conv/Conv2D:output:07stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
!stack_1_block1_se_sigmoid/SigmoidSigmoid)stack_1_block1_se_2_conv/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
stack_1_block1_se_out/mulMul-stack_1_block1_MB_dw_swish/IdentityN:output:0%stack_1_block1_se_sigmoid/Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOpReadVariableOp8stack_1_block1_mb_pw_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
 stack_1_block1_MB_pw_conv/Conv2DConv2Dstack_1_block1_se_out/mul:z:07stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
?
&stack_1_block1_MB_pw_bn/ReadVariableOpReadVariableOp/stack_1_block1_mb_pw_bn_readvariableop_resource*
_output_shapes
:*
dtype0?
(stack_1_block1_MB_pw_bn/ReadVariableOp_1ReadVariableOp1stack_1_block1_mb_pw_bn_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp@stack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBstack_1_block1_mb_pw_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(stack_1_block1_MB_pw_bn/FusedBatchNormV3FusedBatchNormV3)stack_1_block1_MB_pw_conv/Conv2D:output:0.stack_1_block1_MB_pw_bn/ReadVariableOp:value:00stack_1_block1_MB_pw_bn/ReadVariableOp_1:value:0?stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:value:0Astack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
is_training( ?
 stack_1_block1_dropdrop/IdentityIdentity,stack_1_block1_MB_pw_bn/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????`?
stack_1_block1_output/addAddV2,stack_1_block0_MB_pw_bn/FusedBatchNormV3:y:0)stack_1_block1_dropdrop/Identity:output:0*
T0*/
_output_shapes
:?????????`?
post_conv/Conv2D/ReadVariableOpReadVariableOp(post_conv_conv2d_readvariableop_resource*'
_output_shapes
:?
*
dtype0?
post_conv/Conv2DConv2Dstack_1_block1_output/add:z:0'post_conv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`?
*
paddingVALID*
strides
s
post_bn/ReadVariableOpReadVariableOppost_bn_readvariableop_resource*
_output_shapes	
:?
*
dtype0w
post_bn/ReadVariableOp_1ReadVariableOp!post_bn_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
'post_bn/FusedBatchNormV3/ReadVariableOpReadVariableOp0post_bn_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
)post_bn/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp2post_bn_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
post_bn/FusedBatchNormV3FusedBatchNormV3post_conv/Conv2D:output:0post_bn/ReadVariableOp:value:0 post_bn/ReadVariableOp_1:value:0/post_bn/FusedBatchNormV3/ReadVariableOp:value:01post_bn/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????`?
:?
:?
:?
:?
:*
epsilon%o?:*
is_training( v
post_swish/SigmoidSigmoidpost_bn/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????`?
?
post_swish/mulMulpost_bn/FusedBatchNormV3:y:0post_swish/Sigmoid:y:0*
T0*0
_output_shapes
:?????????`?
n
post_swish/IdentityIdentitypost_swish/mul:z:0*
T0*0
_output_shapes
:?????????`?
?
post_swish/IdentityN	IdentityNpost_swish/mul:z:0post_bn/FusedBatchNormV3:y:0*
T
2*+
_gradient_op_typeCustomGradient-45527*L
_output_shapes:
8:?????????`?
:?????????`?
p
avg_pool/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
avg_pool/MeanMeanpost_swish/IdentityN:output:0(avg_pool/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????
i
head_drop/IdentityIdentityavg_pool/Mean:output:0*
T0*(
_output_shapes
:??????????
?
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
predictions/MatMulMatMulhead_drop/Identity:output:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
predictions/SoftmaxSoftmaxpredictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????l
IdentityIdentitypredictions/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp(^post_bn/FusedBatchNormV3/ReadVariableOp*^post_bn/FusedBatchNormV3/ReadVariableOp_1^post_bn/ReadVariableOp^post_bn/ReadVariableOp_1 ^post_conv/Conv2D/ReadVariableOp#^predictions/BiasAdd/ReadVariableOp"^predictions/MatMul/ReadVariableOp/^stack_0_block0_MB_dw_/depthwise/ReadVariableOp8^stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:^stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_0_block0_MB_dw_bn/ReadVariableOp)^stack_0_block0_MB_dw_bn/ReadVariableOp_18^stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:^stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_0_block0_MB_pw_bn/ReadVariableOp)^stack_0_block0_MB_pw_bn/ReadVariableOp_10^stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp0^stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp/^stack_0_block0_se_1_conv/Conv2D/ReadVariableOp0^stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp/^stack_0_block0_se_2_conv/Conv2D/ReadVariableOp/^stack_1_block0_MB_dw_/depthwise/ReadVariableOp8^stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:^stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_1_block0_MB_dw_bn/ReadVariableOp)^stack_1_block0_MB_dw_bn/ReadVariableOp_18^stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:^stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_1_block0_MB_pw_bn/ReadVariableOp)^stack_1_block0_MB_pw_bn/ReadVariableOp_10^stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp0^stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp/^stack_1_block0_se_1_conv/Conv2D/ReadVariableOp0^stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp/^stack_1_block0_se_2_conv/Conv2D/ReadVariableOp/^stack_1_block1_MB_dw_/depthwise/ReadVariableOp8^stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp:^stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_1_block1_MB_dw_bn/ReadVariableOp)^stack_1_block1_MB_dw_bn/ReadVariableOp_18^stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp:^stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_1'^stack_1_block1_MB_pw_bn/ReadVariableOp)^stack_1_block1_MB_pw_bn/ReadVariableOp_10^stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp0^stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp/^stack_1_block1_se_1_conv/Conv2D/ReadVariableOp0^stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp/^stack_1_block1_se_2_conv/Conv2D/ReadVariableOp(^stem_bn/FusedBatchNormV3/ReadVariableOp*^stem_bn/FusedBatchNormV3/ReadVariableOp_1^stem_bn/ReadVariableOp^stem_bn/ReadVariableOp_1 ^stem_conv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'post_bn/FusedBatchNormV3/ReadVariableOp'post_bn/FusedBatchNormV3/ReadVariableOp2V
)post_bn/FusedBatchNormV3/ReadVariableOp_1)post_bn/FusedBatchNormV3/ReadVariableOp_120
post_bn/ReadVariableOppost_bn/ReadVariableOp24
post_bn/ReadVariableOp_1post_bn/ReadVariableOp_12B
post_conv/Conv2D/ReadVariableOppost_conv/Conv2D/ReadVariableOp2H
"predictions/BiasAdd/ReadVariableOp"predictions/BiasAdd/ReadVariableOp2F
!predictions/MatMul/ReadVariableOp!predictions/MatMul/ReadVariableOp2`
.stack_0_block0_MB_dw_/depthwise/ReadVariableOp.stack_0_block0_MB_dw_/depthwise/ReadVariableOp2r
7stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp7stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_19stack_0_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_0_block0_MB_dw_bn/ReadVariableOp&stack_0_block0_MB_dw_bn/ReadVariableOp2T
(stack_0_block0_MB_dw_bn/ReadVariableOp_1(stack_0_block0_MB_dw_bn/ReadVariableOp_12r
7stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp7stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_19stack_0_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_0_block0_MB_pw_bn/ReadVariableOp&stack_0_block0_MB_pw_bn/ReadVariableOp2T
(stack_0_block0_MB_pw_bn/ReadVariableOp_1(stack_0_block0_MB_pw_bn/ReadVariableOp_12b
/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp/stack_0_block0_MB_pw_conv/Conv2D/ReadVariableOp2b
/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp/stack_0_block0_se_1_conv/BiasAdd/ReadVariableOp2`
.stack_0_block0_se_1_conv/Conv2D/ReadVariableOp.stack_0_block0_se_1_conv/Conv2D/ReadVariableOp2b
/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp/stack_0_block0_se_2_conv/BiasAdd/ReadVariableOp2`
.stack_0_block0_se_2_conv/Conv2D/ReadVariableOp.stack_0_block0_se_2_conv/Conv2D/ReadVariableOp2`
.stack_1_block0_MB_dw_/depthwise/ReadVariableOp.stack_1_block0_MB_dw_/depthwise/ReadVariableOp2r
7stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp7stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_19stack_1_block0_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_1_block0_MB_dw_bn/ReadVariableOp&stack_1_block0_MB_dw_bn/ReadVariableOp2T
(stack_1_block0_MB_dw_bn/ReadVariableOp_1(stack_1_block0_MB_dw_bn/ReadVariableOp_12r
7stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp7stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_19stack_1_block0_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_1_block0_MB_pw_bn/ReadVariableOp&stack_1_block0_MB_pw_bn/ReadVariableOp2T
(stack_1_block0_MB_pw_bn/ReadVariableOp_1(stack_1_block0_MB_pw_bn/ReadVariableOp_12b
/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp/stack_1_block0_MB_pw_conv/Conv2D/ReadVariableOp2b
/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp/stack_1_block0_se_1_conv/BiasAdd/ReadVariableOp2`
.stack_1_block0_se_1_conv/Conv2D/ReadVariableOp.stack_1_block0_se_1_conv/Conv2D/ReadVariableOp2b
/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp/stack_1_block0_se_2_conv/BiasAdd/ReadVariableOp2`
.stack_1_block0_se_2_conv/Conv2D/ReadVariableOp.stack_1_block0_se_2_conv/Conv2D/ReadVariableOp2`
.stack_1_block1_MB_dw_/depthwise/ReadVariableOp.stack_1_block1_MB_dw_/depthwise/ReadVariableOp2r
7stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp7stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_19stack_1_block1_MB_dw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_1_block1_MB_dw_bn/ReadVariableOp&stack_1_block1_MB_dw_bn/ReadVariableOp2T
(stack_1_block1_MB_dw_bn/ReadVariableOp_1(stack_1_block1_MB_dw_bn/ReadVariableOp_12r
7stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp7stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp2v
9stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_19stack_1_block1_MB_pw_bn/FusedBatchNormV3/ReadVariableOp_12P
&stack_1_block1_MB_pw_bn/ReadVariableOp&stack_1_block1_MB_pw_bn/ReadVariableOp2T
(stack_1_block1_MB_pw_bn/ReadVariableOp_1(stack_1_block1_MB_pw_bn/ReadVariableOp_12b
/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp/stack_1_block1_MB_pw_conv/Conv2D/ReadVariableOp2b
/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp/stack_1_block1_se_1_conv/BiasAdd/ReadVariableOp2`
.stack_1_block1_se_1_conv/Conv2D/ReadVariableOp.stack_1_block1_se_1_conv/Conv2D/ReadVariableOp2b
/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp/stack_1_block1_se_2_conv/BiasAdd/ReadVariableOp2`
.stack_1_block1_se_2_conv/Conv2D/ReadVariableOp.stack_1_block1_se_2_conv/Conv2D/ReadVariableOp2R
'stem_bn/FusedBatchNormV3/ReadVariableOp'stem_bn/FusedBatchNormV3/ReadVariableOp2V
)stem_bn/FusedBatchNormV3/ReadVariableOp_1)stem_bn/FusedBatchNormV3/ReadVariableOp_120
stem_bn/ReadVariableOpstem_bn/ReadVariableOp24
stem_bn/ReadVariableOp_1stem_bn/ReadVariableOp_12B
stem_conv/Conv2D/ReadVariableOpstem_conv/Conv2D/ReadVariableOp:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?
U
9__inference_stack_1_block1_se_sigmoid_layer_call_fn_46940

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_se_sigmoid_layer_call_and_return_conditional_losses_43214h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_stack_1_block0_MB_pw_conv_layer_call_and_return_conditional_losses_46594

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????`^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_48231
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
p
T__inference_stack_1_block0_se_sigmoid_layer_call_and_return_conditional_losses_43057

inputs
identityT
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????[
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
p
T__inference_stack_1_block0_se_sigmoid_layer_call_and_return_conditional_losses_46568

inputs
identityT
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????[
IdentityIdentitySigmoid:y:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46041

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_44044x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
E
)__inference_head_drop_layer_call_fn_47322

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_head_drop_layer_call_and_return_conditional_losses_43339a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?

?
"__inference_internal_grad_fn_48006
result_grads_0
result_grads_1$
 sigmoid_post_bn_fusedbatchnormv3
identity?
SigmoidSigmoid sigmoid_post_bn_fusedbatchnormv3^result_grads_0*
T0*0
_output_shapes
:?????????`?
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????`?
p
mulMul sigmoid_post_bn_fusedbatchnormv3sub:z:0*
T0*0
_output_shapes
:?????????`?
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????`?
]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????`?
b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????`?
Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????`?
"
identityIdentity:output:0*g
_input_shapesV
T:?????????`?
:?????????`?
:?????????`?
:` \
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????`?

?
?
"__inference_internal_grad_fn_48111
result_grads_0
result_grads_1,
(sigmoid_stack_1_block1_se_1_conv_biasadd
identity?
SigmoidSigmoid(sigmoid_stack_1_block1_se_1_conv_biasadd^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????w
mulMul(sigmoid_stack_1_block1_se_1_conv_biasaddsub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
?
T__inference_stack_0_block0_MB_pw_conv_layer_call_and_return_conditional_losses_46217

inputs8
conv2d_readvariableop_resource: 
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????? : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
T__inference_stack_0_block0_MB_pw_conv_layer_call_and_return_conditional_losses_42917

inputs8
conv2d_readvariableop_resource: 
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????? : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47931
result_grads_0
result_grads_1,
(sigmoid_stack_0_block0_se_1_conv_biasadd
identity?
SigmoidSigmoid(sigmoid_stack_0_block0_se_1_conv_biasadd^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????w
mulMul(sigmoid_stack_0_block0_se_1_conv_biasaddsub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43792

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
T__inference_stack_1_block0_MB_pw_conv_layer_call_and_return_conditional_losses_43074

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????`*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????`^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
c
E__inference_stem_swish_layer_call_and_return_conditional_losses_42795

inputs

identity_1U
SigmoidSigmoidinputs*
T0*0
_output_shapes
:?????????? Z
mulMulinputsSigmoid:y:0*
T0*0
_output_shapes
:?????????? X
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:?????????? ?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-42788*L
_output_shapes:
8:?????????? :?????????? e

Identity_1IdentityIdentityN:output:0*
T0*0
_output_shapes
:?????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
|
P__inference_stack_0_block0_se_out_layer_call_and_return_conditional_losses_46203
inputs_0
inputs_1
identityY
mulMulinputs_0inputs_1*
T0*0
_output_shapes
:?????????? X
IdentityIdentitymul:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????? :????????? :Z V
0
_output_shapes
:?????????? 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?

?
S__inference_stack_0_block0_se_2_conv_layer_call_and_return_conditional_losses_46181

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
q
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_43601

inputs
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskJ
packed/1Const*
_output_shapes
: *
dtype0*
value	B :J
packed/2Const*
_output_shapes
: *
dtype0*
value	B :J
packed/3Const*
_output_shapes
: *
dtype0*
value	B :?
packedPackstrided_slice:output:0packed/1:output:0packed/2:output:0packed/3:output:0*
N*
T0*
_output_shapes
:R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????`?
$dropout/random_uniform/RandomUniformRandomUniformpacked:output:0*
T0*/
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??*>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????`a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46646

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_43792w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
s
U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_46882

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????`Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????`W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????`?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-46875*J
_output_shapes8
6:?????????`:?????????`d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
l
P__inference_stack_1_block0_output_layer_call_and_return_conditional_losses_46727

inputs
identityV
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
,__inference_EfficientNet_layer_call_fn_45290

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: #
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9: 

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:

unknown_15:

unknown_16:

unknown_17:$

unknown_18:

unknown_19:

unknown_20:

unknown_21:

unknown_22:$

unknown_23:

unknown_24:$

unknown_25:

unknown_26:$

unknown_27:

unknown_28:

unknown_29:

unknown_30:

unknown_31:$

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:$

unknown_37:

unknown_38:$

unknown_39:

unknown_40:$

unknown_41:

unknown_42:

unknown_43:

unknown_44:

unknown_45:%

unknown_46:?


unknown_47:	?


unknown_48:	?


unknown_49:	?


unknown_50:	?


unknown_51:	?


unknown_52:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*H
_read_only_resource_inputs*
(&"#$'()*+,-01256*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_EfficientNet_layer_call_and_return_conditional_losses_44397o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:????????? ?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:????????? ?
 
_user_specified_nameinputs
?	
?
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46756

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_42557?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47691
result_grads_0
result_grads_19
5sigmoid_efficientnet_stack_0_block0_se_1_conv_biasadd
identity?
SigmoidSigmoid5sigmoid_efficientnet_stack_0_block0_se_1_conv_biasadd^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:??????????
mulMul5sigmoid_efficientnet_stack_0_block0_se_1_conv_biasaddsub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
?
B__inference_stem_bn_layer_call_and_return_conditional_losses_42775

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*L
_output_shapes:
8:?????????? : : : : :*
epsilon%o?:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
8__inference_stack_1_block1_se_2_conv_layer_call_fn_46925

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_stack_1_block1_se_2_conv_layer_call_and_return_conditional_losses_43203w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_post_conv_layer_call_and_return_conditional_losses_47156

inputs9
conv2d_readvariableop_resource:?

identity??Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?
*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`?
*
paddingVALID*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????`?
^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_48171
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?
?
D__inference_post_conv_layer_call_and_return_conditional_losses_43284

inputs9
conv2d_readvariableop_resource:?

identity??Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?
*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????`?
*
paddingVALID*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????`?
^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47946
result_grads_0
result_grads_14
0sigmoid_stack_1_block0_mb_dw_bn_fusedbatchnormv3
identity?
SigmoidSigmoid0sigmoid_stack_1_block0_mb_dw_bn_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`
mulMul0sigmoid_stack_1_block0_mb_dw_bn_fusedbatchnormv3sub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?
_
C__inference_avg_pool_layer_call_and_return_conditional_losses_47317

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      h
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:??????????
V
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:??????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????`?
:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?
?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42301

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
B__inference_post_bn_layer_call_and_return_conditional_losses_42685

inputs&
readvariableop_resource:	?
(
readvariableop_1_resource:	?
7
(fusedbatchnormv3_readvariableop_resource:	?
9
*fusedbatchnormv3_readvariableop_1_resource:	?

identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?
*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?
*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?
*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????
:?
:?
:?
:?
:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????
?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????
: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????

 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47916
result_grads_0
result_grads_14
0sigmoid_stack_0_block0_mb_dw_bn_fusedbatchnormv3
identity?
SigmoidSigmoid0sigmoid_stack_0_block0_mb_dw_bn_fusedbatchnormv3^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ?
mulMul0sigmoid_stack_0_block0_mb_dw_bn_fusedbatchnormv3sub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
?
?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_42621

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_48066
result_grads_0
result_grads_14
0sigmoid_stack_1_block0_mb_dw_bn_fusedbatchnormv3
identity?
SigmoidSigmoid0sigmoid_stack_1_block0_mb_dw_bn_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`
mulMul0sigmoid_stack_1_block0_mb_dw_bn_fusedbatchnormv3sub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?
|
P__inference_stack_1_block1_output_layer_call_and_return_conditional_losses_47142
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:?????????`W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????`:?????????`:Y U
/
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????`
"
_user_specified_name
inputs/1
?	
?
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46015

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42332?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
5__inference_stack_0_block0_MB_dw__layer_call_fn_45980

inputs!
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_stack_0_block0_MB_dw__layer_call_and_return_conditional_losses_42806x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????? : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
|
P__inference_stack_1_block1_se_out_layer_call_and_return_conditional_losses_46957
inputs_0
inputs_1
identityX
mulMulinputs_0inputs_1*
T0*/
_output_shapes
:?????????`W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????`:?????????:Y U
/
_output_shapes
:?????????`
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

n
"__inference_internal_grad_fn_48246
result_grads_0
result_grads_1
sigmoid_inputs
identityn
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*0
_output_shapes
:?????????`?
J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????`?
^
mulMulsigmoid_inputssub:z:0*
T0*0
_output_shapes
:?????????`?
J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????`?
]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????`?
b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????`?
Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????`?
"
identityIdentity:output:0*g
_input_shapesV
T:?????????`?
:?????????`?
:?????????`?
:` \
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????`?

(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????`?

?
b
D__inference_head_drop_layer_call_and_return_conditional_losses_43339

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????
\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????
"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????
:P L
(
_output_shapes
:??????????

 
_user_specified_nameinputs
?
?
B__inference_stem_bn_layer_call_and_return_conditional_losses_45904

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
9__inference_stack_1_block1_MB_pw_conv_layer_call_fn_46964

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block1_MB_pw_conv_layer_call_and_return_conditional_losses_43231w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????`: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46782

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_43141w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_47871
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*d
_input_shapesS
Q:?????????:?????????:?????????:_ [
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????
?

n
"__inference_internal_grad_fn_48216
result_grads_0
result_grads_1
sigmoid_inputs
identitym
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`]
mulMulsigmoid_inputssub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?	
?
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46769

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_42588?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

n
"__inference_internal_grad_fn_48141
result_grads_0
result_grads_1
sigmoid_inputs
identityn
SigmoidSigmoidsigmoid_inputs^result_grads_0*
T0*0
_output_shapes
:?????????? J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??b
subSubsub/x:output:0Sigmoid:y:0*
T0*0
_output_shapes
:?????????? ^
mulMulsigmoid_inputssub:z:0*
T0*0
_output_shapes
:?????????? J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??`
addAddV2add/x:output:0mul:z:0*
T0*0
_output_shapes
:?????????? ]
mul_1MulSigmoid:y:0add:z:0*
T0*0
_output_shapes
:?????????? b
mul_2Mulresult_grads_0	mul_1:z:0*
T0*0
_output_shapes
:?????????? Z
IdentityIdentity	mul_2:z:0*
T0*0
_output_shapes
:?????????? "
identityIdentity:output:0*g
_input_shapesV
T:?????????? :?????????? :?????????? :` \
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_0:`\
0
_output_shapes
:?????????? 
(
_user_specified_nameresult_grads_1:62
0
_output_shapes
:?????????? 
?
F
*__inference_post_swish_layer_call_fn_47285

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????`?
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_post_swish_layer_call_and_return_conditional_losses_43325i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????`?
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????`?
:X T
0
_output_shapes
:?????????`?

 
_user_specified_nameinputs
?	
?
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46392

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *[
fVRT
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_42460?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_internal_grad_fn_47706
result_grads_0
result_grads_1A
=sigmoid_efficientnet_stack_1_block0_mb_dw_bn_fusedbatchnormv3
identity?
SigmoidSigmoid=sigmoid_efficientnet_stack_1_block0_mb_dw_bn_fusedbatchnormv3^result_grads_0*
T0*/
_output_shapes
:?????????`J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??a
subSubsub/x:output:0Sigmoid:y:0*
T0*/
_output_shapes
:?????????`?
mulMul=sigmoid_efficientnet_stack_1_block0_mb_dw_bn_fusedbatchnormv3sub:z:0*
T0*/
_output_shapes
:?????????`J
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??_
addAddV2add/x:output:0mul:z:0*
T0*/
_output_shapes
:?????????`\
mul_1MulSigmoid:y:0add:z:0*
T0*/
_output_shapes
:?????????`a
mul_2Mulresult_grads_0	mul_1:z:0*
T0*/
_output_shapes
:?????????`Y
IdentityIdentity	mul_2:z:0*
T0*/
_output_shapes
:?????????`"
identityIdentity:output:0*d
_input_shapesS
Q:?????????`:?????????`:?????????`:_ [
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_0:_[
/
_output_shapes
:?????????`
(
_user_specified_nameresult_grads_1:51
/
_output_shapes
:?????????`
?
s
U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_43004

inputs

identity_1T
SigmoidSigmoidinputs*
T0*/
_output_shapes
:?????????`Y
mulMulinputsSigmoid:y:0*
T0*/
_output_shapes
:?????????`W
IdentityIdentitymul:z:0*
T0*/
_output_shapes
:?????????`?
	IdentityN	IdentityNmul:z:0inputs*
T
2*+
_gradient_op_typeCustomGradient-42997*J
_output_shapes8
6:?????????`:?????????`d

Identity_1IdentityIdentityN:output:0*
T0*/
_output_shapes
:?????????`"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????`:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
U
9__inference_stack_1_block0_se_sigmoid_layer_call_fn_46563

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_stack_1_block0_se_sigmoid_layer_call_and_return_conditional_losses_43057h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47095

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????`:::::*
epsilon%o?:*
exponential_avg_factor%???=?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????`?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_42493

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:
"__inference_internal_grad_fn_47661CustomGradient-41981:
"__inference_internal_grad_fn_47676CustomGradient-42006:
"__inference_internal_grad_fn_47691CustomGradient-42020:
"__inference_internal_grad_fn_47706CustomGradient-42070:
"__inference_internal_grad_fn_47721CustomGradient-42084:
"__inference_internal_grad_fn_47736CustomGradient-42134:
"__inference_internal_grad_fn_47751CustomGradient-42148:
"__inference_internal_grad_fn_47766CustomGradient-42198:
"__inference_internal_grad_fn_47781CustomGradient-42788:
"__inference_internal_grad_fn_47796CustomGradient-42840:
"__inference_internal_grad_fn_47811CustomGradient-42870:
"__inference_internal_grad_fn_47826CustomGradient-42997:
"__inference_internal_grad_fn_47841CustomGradient-43027:
"__inference_internal_grad_fn_47856CustomGradient-43154:
"__inference_internal_grad_fn_47871CustomGradient-43184:
"__inference_internal_grad_fn_47886CustomGradient-43318:
"__inference_internal_grad_fn_47901CustomGradient-45310:
"__inference_internal_grad_fn_47916CustomGradient-45335:
"__inference_internal_grad_fn_47931CustomGradient-45349:
"__inference_internal_grad_fn_47946CustomGradient-45399:
"__inference_internal_grad_fn_47961CustomGradient-45413:
"__inference_internal_grad_fn_47976CustomGradient-45463:
"__inference_internal_grad_fn_47991CustomGradient-45477:
"__inference_internal_grad_fn_48006CustomGradient-45527:
"__inference_internal_grad_fn_48021CustomGradient-45564:
"__inference_internal_grad_fn_48036CustomGradient-45589:
"__inference_internal_grad_fn_48051CustomGradient-45603:
"__inference_internal_grad_fn_48066CustomGradient-45653:
"__inference_internal_grad_fn_48081CustomGradient-45667:
"__inference_internal_grad_fn_48096CustomGradient-45717:
"__inference_internal_grad_fn_48111CustomGradient-45731:
"__inference_internal_grad_fn_48126CustomGradient-45796:
"__inference_internal_grad_fn_48141CustomGradient-45966:
"__inference_internal_grad_fn_48156CustomGradient-46121:
"__inference_internal_grad_fn_48171CustomGradient-46155:
"__inference_internal_grad_fn_48186CustomGradient-46498:
"__inference_internal_grad_fn_48201CustomGradient-46532:
"__inference_internal_grad_fn_48216CustomGradient-46875:
"__inference_internal_grad_fn_48231CustomGradient-46909:
"__inference_internal_grad_fn_48246CustomGradient-47288"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
D
input_19
serving_default_input_1:0????????? ??
predictions0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer_with_weights-8
layer-16
layer_with_weights-9
layer-17
layer-18
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer-24
layer_with_weights-12
layer-25
layer_with_weights-13
layer-26
layer-27
layer_with_weights-14
layer-28
layer_with_weights-15
layer-29
layer-30
 layer-31
!layer_with_weights-16
!layer-32
"layer-33
#layer_with_weights-17
#layer-34
$layer-35
%layer-36
&layer_with_weights-18
&layer-37
'layer_with_weights-19
'layer-38
(layer-39
)layer-40
*layer_with_weights-20
*layer-41
+layer_with_weights-21
+layer-42
,layer-43
-layer-44
.layer-45
/layer_with_weights-22
/layer-46
0	optimizer
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

6kernel
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Hdepthwise_kernel
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
(
Z	keras_api"
_tf_keras_layer
?

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ekernel
fbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

skernel
t	variables
utrainable_variables
vregularization_losses
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
xaxis
	ygamma
zbeta
{moving_mean
|moving_variance
}	variables
~trainable_variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?depthwise_kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?depthwise_kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
)
?	keras_api"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?noise_shape
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?beta_1
?beta_2

?decay
?learning_rate
	?iter6m?<m?=m?Hm?Nm?Om?[m?\m?em?fm?sm?ym?zm?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?6v?<v?=v?Hv?Nv?Ov?[v?\v?ev?fv?sv?yv?zv?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
?
60
<1
=2
>3
?4
H5
N6
O7
P8
Q9
[10
\11
e12
f13
s14
y15
z16
{17
|18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53"
trackable_list_wrapper
?
60
<1
=2
H3
N4
O5
[6
\7
e8
f9
s10
y11
z12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
1	variables
2trainable_variables
3regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2stem_conv/kernel
'
60"
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2stem_bn/gamma
: 2stem_bn/beta
#:!  (2stem_bn/moving_mean
':%  (2stem_bn/moving_variance
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
@:> 2&stack_0_block0_MB_dw_/depthwise_kernel
'
H0"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:) 2stack_0_block0_MB_dw_bn/gamma
*:( 2stack_0_block0_MB_dw_bn/beta
3:1  (2#stack_0_block0_MB_dw_bn/moving_mean
7:5  (2'stack_0_block0_MB_dw_bn/moving_variance
<
N0
O1
P2
Q3"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
9:7 2stack_0_block0_se_1_conv/kernel
+:)2stack_0_block0_se_1_conv/bias
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
]	variables
^trainable_variables
_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
9:7 2stack_0_block0_se_2_conv/kernel
+:) 2stack_0_block0_se_2_conv/bias
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
k	variables
ltrainable_variables
mregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
::8 2 stack_0_block0_MB_pw_conv/kernel
'
s0"
trackable_list_wrapper
'
s0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
t	variables
utrainable_variables
vregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2stack_0_block0_MB_pw_bn/gamma
*:(2stack_0_block0_MB_pw_bn/beta
3:1 (2#stack_0_block0_MB_pw_bn/moving_mean
7:5 (2'stack_0_block0_MB_pw_bn/moving_variance
<
y0
z1
{2
|3"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
}	variables
~trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
@:>2&stack_1_block0_MB_dw_/depthwise_kernel
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2stack_1_block0_MB_dw_bn/gamma
*:(2stack_1_block0_MB_dw_bn/beta
3:1 (2#stack_1_block0_MB_dw_bn/moving_mean
7:5 (2'stack_1_block0_MB_dw_bn/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
9:72stack_1_block0_se_1_conv/kernel
+:)2stack_1_block0_se_1_conv/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
9:72stack_1_block0_se_2_conv/kernel
+:)2stack_1_block0_se_2_conv/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
::82 stack_1_block0_MB_pw_conv/kernel
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2stack_1_block0_MB_pw_bn/gamma
*:(2stack_1_block0_MB_pw_bn/beta
3:1 (2#stack_1_block0_MB_pw_bn/moving_mean
7:5 (2'stack_1_block0_MB_pw_bn/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
@:>2&stack_1_block1_MB_dw_/depthwise_kernel
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2stack_1_block1_MB_dw_bn/gamma
*:(2stack_1_block1_MB_dw_bn/beta
3:1 (2#stack_1_block1_MB_dw_bn/moving_mean
7:5 (2'stack_1_block1_MB_dw_bn/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
9:72stack_1_block1_se_1_conv/kernel
+:)2stack_1_block1_se_1_conv/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
9:72stack_1_block1_se_2_conv/kernel
+:)2stack_1_block1_se_2_conv/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
::82 stack_1_block1_MB_pw_conv/kernel
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
+:)2stack_1_block1_MB_pw_bn/gamma
*:(2stack_1_block1_MB_pw_bn/beta
3:1 (2#stack_1_block1_MB_pw_bn/moving_mean
7:5 (2'stack_1_block1_MB_pw_bn/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?
2post_conv/kernel
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?
2post_bn/gamma
:?
2post_bn/beta
$:"?
 (2post_bn/moving_mean
(:&?
 (2post_bn/moving_variance
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#	?
2predictions/kernel
:2predictions/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
?
>0
?1
P2
Q3
{4
|5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15"
trackable_list_wrapper
?
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
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
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
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:- 2Adam/stem_conv/kernel/m
 : 2Adam/stem_bn/gamma/m
: 2Adam/stem_bn/beta/m
E:C 2-Adam/stack_0_block0_MB_dw_/depthwise_kernel/m
0:. 2$Adam/stack_0_block0_MB_dw_bn/gamma/m
/:- 2#Adam/stack_0_block0_MB_dw_bn/beta/m
>:< 2&Adam/stack_0_block0_se_1_conv/kernel/m
0:.2$Adam/stack_0_block0_se_1_conv/bias/m
>:< 2&Adam/stack_0_block0_se_2_conv/kernel/m
0:. 2$Adam/stack_0_block0_se_2_conv/bias/m
?:= 2'Adam/stack_0_block0_MB_pw_conv/kernel/m
0:.2$Adam/stack_0_block0_MB_pw_bn/gamma/m
/:-2#Adam/stack_0_block0_MB_pw_bn/beta/m
E:C2-Adam/stack_1_block0_MB_dw_/depthwise_kernel/m
0:.2$Adam/stack_1_block0_MB_dw_bn/gamma/m
/:-2#Adam/stack_1_block0_MB_dw_bn/beta/m
>:<2&Adam/stack_1_block0_se_1_conv/kernel/m
0:.2$Adam/stack_1_block0_se_1_conv/bias/m
>:<2&Adam/stack_1_block0_se_2_conv/kernel/m
0:.2$Adam/stack_1_block0_se_2_conv/bias/m
?:=2'Adam/stack_1_block0_MB_pw_conv/kernel/m
0:.2$Adam/stack_1_block0_MB_pw_bn/gamma/m
/:-2#Adam/stack_1_block0_MB_pw_bn/beta/m
E:C2-Adam/stack_1_block1_MB_dw_/depthwise_kernel/m
0:.2$Adam/stack_1_block1_MB_dw_bn/gamma/m
/:-2#Adam/stack_1_block1_MB_dw_bn/beta/m
>:<2&Adam/stack_1_block1_se_1_conv/kernel/m
0:.2$Adam/stack_1_block1_se_1_conv/bias/m
>:<2&Adam/stack_1_block1_se_2_conv/kernel/m
0:.2$Adam/stack_1_block1_se_2_conv/bias/m
?:=2'Adam/stack_1_block1_MB_pw_conv/kernel/m
0:.2$Adam/stack_1_block1_MB_pw_bn/gamma/m
/:-2#Adam/stack_1_block1_MB_pw_bn/beta/m
0:.?
2Adam/post_conv/kernel/m
!:?
2Adam/post_bn/gamma/m
 :?
2Adam/post_bn/beta/m
*:(	?
2Adam/predictions/kernel/m
#:!2Adam/predictions/bias/m
/:- 2Adam/stem_conv/kernel/v
 : 2Adam/stem_bn/gamma/v
: 2Adam/stem_bn/beta/v
E:C 2-Adam/stack_0_block0_MB_dw_/depthwise_kernel/v
0:. 2$Adam/stack_0_block0_MB_dw_bn/gamma/v
/:- 2#Adam/stack_0_block0_MB_dw_bn/beta/v
>:< 2&Adam/stack_0_block0_se_1_conv/kernel/v
0:.2$Adam/stack_0_block0_se_1_conv/bias/v
>:< 2&Adam/stack_0_block0_se_2_conv/kernel/v
0:. 2$Adam/stack_0_block0_se_2_conv/bias/v
?:= 2'Adam/stack_0_block0_MB_pw_conv/kernel/v
0:.2$Adam/stack_0_block0_MB_pw_bn/gamma/v
/:-2#Adam/stack_0_block0_MB_pw_bn/beta/v
E:C2-Adam/stack_1_block0_MB_dw_/depthwise_kernel/v
0:.2$Adam/stack_1_block0_MB_dw_bn/gamma/v
/:-2#Adam/stack_1_block0_MB_dw_bn/beta/v
>:<2&Adam/stack_1_block0_se_1_conv/kernel/v
0:.2$Adam/stack_1_block0_se_1_conv/bias/v
>:<2&Adam/stack_1_block0_se_2_conv/kernel/v
0:.2$Adam/stack_1_block0_se_2_conv/bias/v
?:=2'Adam/stack_1_block0_MB_pw_conv/kernel/v
0:.2$Adam/stack_1_block0_MB_pw_bn/gamma/v
/:-2#Adam/stack_1_block0_MB_pw_bn/beta/v
E:C2-Adam/stack_1_block1_MB_dw_/depthwise_kernel/v
0:.2$Adam/stack_1_block1_MB_dw_bn/gamma/v
/:-2#Adam/stack_1_block1_MB_dw_bn/beta/v
>:<2&Adam/stack_1_block1_se_1_conv/kernel/v
0:.2$Adam/stack_1_block1_se_1_conv/bias/v
>:<2&Adam/stack_1_block1_se_2_conv/kernel/v
0:.2$Adam/stack_1_block1_se_2_conv/bias/v
?:=2'Adam/stack_1_block1_MB_pw_conv/kernel/v
0:.2$Adam/stack_1_block1_MB_pw_bn/gamma/v
/:-2#Adam/stack_1_block1_MB_pw_bn/beta/v
0:.?
2Adam/post_conv/kernel/v
!:?
2Adam/post_bn/gamma/v
 :?
2Adam/post_bn/beta/v
*:(	?
2Adam/predictions/kernel/v
#:!2Adam/predictions/bias/v
?2?
,__inference_EfficientNet_layer_call_fn_43470
,__inference_EfficientNet_layer_call_fn_45177
,__inference_EfficientNet_layer_call_fn_45290
,__inference_EfficientNet_layer_call_fn_44621?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544
G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820
G__inference_EfficientNet_layer_call_and_return_conditional_losses_44782
G__inference_EfficientNet_layer_call_and_return_conditional_losses_44943?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_42215input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_stem_conv_layer_call_fn_45827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_stem_conv_layer_call_and_return_conditional_losses_45834?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_stem_bn_layer_call_fn_45847
'__inference_stem_bn_layer_call_fn_45860
'__inference_stem_bn_layer_call_fn_45873
'__inference_stem_bn_layer_call_fn_45886?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_stem_bn_layer_call_and_return_conditional_losses_45904
B__inference_stem_bn_layer_call_and_return_conditional_losses_45922
B__inference_stem_bn_layer_call_and_return_conditional_losses_45940
B__inference_stem_bn_layer_call_and_return_conditional_losses_45958?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_stem_swish_layer_call_fn_45963?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_stem_swish_layer_call_and_return_conditional_losses_45973?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack_0_block0_MB_dw__layer_call_fn_45980?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack_0_block0_MB_dw__layer_call_and_return_conditional_losses_45989?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46002
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46015
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46028
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46041?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46059
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46077
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46095
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46113?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
:__inference_stack_0_block0_MB_dw_swish_layer_call_fn_46118?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_46128?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_stack_0_block0_se_1_conv_layer_call_fn_46137?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_stack_0_block0_se_1_conv_layer_call_and_return_conditional_losses_46147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_stack_0_block0_se_swish_layer_call_fn_46152?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_46162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_stack_0_block0_se_2_conv_layer_call_fn_46171?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_stack_0_block0_se_2_conv_layer_call_and_return_conditional_losses_46181?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_stack_0_block0_se_sigmoid_layer_call_fn_46186?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_stack_0_block0_se_sigmoid_layer_call_and_return_conditional_losses_46191?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack_0_block0_se_out_layer_call_fn_46197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack_0_block0_se_out_layer_call_and_return_conditional_losses_46203?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_stack_0_block0_MB_pw_conv_layer_call_fn_46210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_stack_0_block0_MB_pw_conv_layer_call_and_return_conditional_losses_46217?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46230
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46243
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46256
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46269?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46287
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46305
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46323
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46341?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_stack_0_block0_output_layer_call_fn_46346?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack_0_block0_output_layer_call_and_return_conditional_losses_46350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack_1_block0_MB_dw__layer_call_fn_46357?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack_1_block0_MB_dw__layer_call_and_return_conditional_losses_46366?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46379
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46392
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46405
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46418?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46436
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46454
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46472
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46490?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
:__inference_stack_1_block0_MB_dw_swish_layer_call_fn_46495?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_46505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_stack_1_block0_se_1_conv_layer_call_fn_46514?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_stack_1_block0_se_1_conv_layer_call_and_return_conditional_losses_46524?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_stack_1_block0_se_swish_layer_call_fn_46529?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_46539?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_stack_1_block0_se_2_conv_layer_call_fn_46548?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_stack_1_block0_se_2_conv_layer_call_and_return_conditional_losses_46558?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_stack_1_block0_se_sigmoid_layer_call_fn_46563?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_stack_1_block0_se_sigmoid_layer_call_and_return_conditional_losses_46568?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack_1_block0_se_out_layer_call_fn_46574?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack_1_block0_se_out_layer_call_and_return_conditional_losses_46580?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_stack_1_block0_MB_pw_conv_layer_call_fn_46587?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_stack_1_block0_MB_pw_conv_layer_call_and_return_conditional_losses_46594?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46607
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46620
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46633
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46646?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46664
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46682
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46700
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46718?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_stack_1_block0_output_layer_call_fn_46723?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack_1_block0_output_layer_call_and_return_conditional_losses_46727?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack_1_block1_MB_dw__layer_call_fn_46734?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack_1_block1_MB_dw__layer_call_and_return_conditional_losses_46743?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46756
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46769
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46782
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46795?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46813
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46831
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46849
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46867?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
:__inference_stack_1_block1_MB_dw_swish_layer_call_fn_46872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_46882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_stack_1_block1_se_1_conv_layer_call_fn_46891?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_stack_1_block1_se_1_conv_layer_call_and_return_conditional_losses_46901?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_stack_1_block1_se_swish_layer_call_fn_46906?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_46916?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_stack_1_block1_se_2_conv_layer_call_fn_46925?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_stack_1_block1_se_2_conv_layer_call_and_return_conditional_losses_46935?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_stack_1_block1_se_sigmoid_layer_call_fn_46940?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_stack_1_block1_se_sigmoid_layer_call_and_return_conditional_losses_46945?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack_1_block1_se_out_layer_call_fn_46951?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack_1_block1_se_out_layer_call_and_return_conditional_losses_46957?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_stack_1_block1_MB_pw_conv_layer_call_fn_46964?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
T__inference_stack_1_block1_MB_pw_conv_layer_call_and_return_conditional_losses_46971?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_46984
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_46997
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_47010
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_47023?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47041
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47059
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47077
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47095?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
7__inference_stack_1_block1_dropdrop_layer_call_fn_47100
7__inference_stack_1_block1_dropdrop_layer_call_fn_47105?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_47110
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_47130?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_stack_1_block1_output_layer_call_fn_47136?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack_1_block1_output_layer_call_and_return_conditional_losses_47142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_post_conv_layer_call_fn_47149?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_post_conv_layer_call_and_return_conditional_losses_47156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_post_bn_layer_call_fn_47169
'__inference_post_bn_layer_call_fn_47182
'__inference_post_bn_layer_call_fn_47195
'__inference_post_bn_layer_call_fn_47208?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_post_bn_layer_call_and_return_conditional_losses_47226
B__inference_post_bn_layer_call_and_return_conditional_losses_47244
B__inference_post_bn_layer_call_and_return_conditional_losses_47262
B__inference_post_bn_layer_call_and_return_conditional_losses_47280?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_post_swish_layer_call_fn_47285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_post_swish_layer_call_and_return_conditional_losses_47295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_avg_pool_layer_call_fn_47300
(__inference_avg_pool_layer_call_fn_47305?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_avg_pool_layer_call_and_return_conditional_losses_47311
C__inference_avg_pool_layer_call_and_return_conditional_losses_47317?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_head_drop_layer_call_fn_47322
)__inference_head_drop_layer_call_fn_47327?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_head_drop_layer_call_and_return_conditional_losses_47332
D__inference_head_drop_layer_call_and_return_conditional_losses_47344?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_predictions_layer_call_fn_47353?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_predictions_layer_call_and_return_conditional_losses_47364?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_45064input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
MbK
'EfficientNet/stem_bn/FusedBatchNormV3:0 __inference__wrapped_model_42215
]b[
7EfficientNet/stack_0_block0_MB_dw_bn/FusedBatchNormV3:0 __inference__wrapped_model_42215
UbS
/EfficientNet/stack_0_block0_se_1_conv/BiasAdd:0 __inference__wrapped_model_42215
]b[
7EfficientNet/stack_1_block0_MB_dw_bn/FusedBatchNormV3:0 __inference__wrapped_model_42215
UbS
/EfficientNet/stack_1_block0_se_1_conv/BiasAdd:0 __inference__wrapped_model_42215
]b[
7EfficientNet/stack_1_block1_MB_dw_bn/FusedBatchNormV3:0 __inference__wrapped_model_42215
UbS
/EfficientNet/stack_1_block1_se_1_conv/BiasAdd:0 __inference__wrapped_model_42215
MbK
'EfficientNet/post_bn/FusedBatchNormV3:0 __inference__wrapped_model_42215
SbQ
inputs:0E__inference_stem_swish_layer_call_and_return_conditional_losses_42795
cba
inputs:0U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_42847
`b^
inputs:0R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_42877
cba
inputs:0U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_43004
`b^
inputs:0R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_43034
cba
inputs:0U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_43161
`b^
inputs:0R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_43191
SbQ
inputs:0E__inference_post_swish_layer_call_and_return_conditional_losses_43325
gbe
stem_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544
wbu
*stack_0_block0_MB_dw_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544
obm
"stack_0_block0_se_1_conv/BiasAdd:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544
wbu
*stack_1_block0_MB_dw_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544
obm
"stack_1_block0_se_1_conv/BiasAdd:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544
wbu
*stack_1_block1_MB_dw_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544
obm
"stack_1_block1_se_1_conv/BiasAdd:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544
gbe
post_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544
gbe
stem_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820
wbu
*stack_0_block0_MB_dw_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820
obm
"stack_0_block0_se_1_conv/BiasAdd:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820
wbu
*stack_1_block0_MB_dw_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820
obm
"stack_1_block0_se_1_conv/BiasAdd:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820
wbu
*stack_1_block1_MB_dw_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820
obm
"stack_1_block1_se_1_conv/BiasAdd:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820
gbe
post_bn/FusedBatchNormV3:0G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820
SbQ
inputs:0E__inference_stem_swish_layer_call_and_return_conditional_losses_45973
cba
inputs:0U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_46128
`b^
inputs:0R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_46162
cba
inputs:0U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_46505
`b^
inputs:0R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_46539
cba
inputs:0U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_46882
`b^
inputs:0R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_46916
SbQ
inputs:0E__inference_post_swish_layer_call_and_return_conditional_losses_47295?
G__inference_EfficientNet_layer_call_and_return_conditional_losses_44782?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????A?>
7?4
*?'
input_1????????? ?
p 

 
? "%?"
?
0?????????
? ?
G__inference_EfficientNet_layer_call_and_return_conditional_losses_44943?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????A?>
7?4
*?'
input_1????????? ?
p

 
? "%?"
?
0?????????
? ?
G__inference_EfficientNet_layer_call_and_return_conditional_losses_45544?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????@?=
6?3
)?&
inputs????????? ?
p 

 
? "%?"
?
0?????????
? ?
G__inference_EfficientNet_layer_call_and_return_conditional_losses_45820?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????@?=
6?3
)?&
inputs????????? ?
p

 
? "%?"
?
0?????????
? ?
,__inference_EfficientNet_layer_call_fn_43470?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????A?>
7?4
*?'
input_1????????? ?
p 

 
? "???????????
,__inference_EfficientNet_layer_call_fn_44621?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????A?>
7?4
*?'
input_1????????? ?
p

 
? "???????????
,__inference_EfficientNet_layer_call_fn_45177?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????@?=
6?3
)?&
inputs????????? ?
p 

 
? "???????????
,__inference_EfficientNet_layer_call_fn_45290?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????@?=
6?3
)?&
inputs????????? ?
p

 
? "???????????
 __inference__wrapped_model_42215?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????9?6
/?,
*?'
input_1????????? ?
? "9?6
4
predictions%?"
predictions??????????
C__inference_avg_pool_layer_call_and_return_conditional_losses_47311?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
C__inference_avg_pool_layer_call_and_return_conditional_losses_47317b8?5
.?+
)?&
inputs?????????`?

? "&?#
?
0??????????

? ?
(__inference_avg_pool_layer_call_fn_47300wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
(__inference_avg_pool_layer_call_fn_47305U8?5
.?+
)?&
inputs?????????`?

? "???????????
?
D__inference_head_drop_layer_call_and_return_conditional_losses_47332^4?1
*?'
!?
inputs??????????

p 
? "&?#
?
0??????????

? ?
D__inference_head_drop_layer_call_and_return_conditional_losses_47344^4?1
*?'
!?
inputs??????????

p
? "&?#
?
0??????????

? ~
)__inference_head_drop_layer_call_fn_47322Q4?1
*?'
!?
inputs??????????

p 
? "???????????
~
)__inference_head_drop_layer_call_fn_47327Q4?1
*?'
!?
inputs??????????

p
? "???????????
?
"__inference_internal_grad_fn_47661??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_47676??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_47691??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_47706??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_47721??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_47736??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_47751??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_47766??w?t
m?j

 
1?.
result_grads_0?????????`?

1?.
result_grads_1?????????`?

? "-?*

 
$?!
1?????????`?
?
"__inference_internal_grad_fn_47781??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_47796??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_47811??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_47826??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_47841??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_47856??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_47871??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_47886??w?t
m?j

 
1?.
result_grads_0?????????`?

1?.
result_grads_1?????????`?

? "-?*

 
$?!
1?????????`?
?
"__inference_internal_grad_fn_47901??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_47916??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_47931??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_47946??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_47961??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_47976??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_47991??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_48006??w?t
m?j

 
1?.
result_grads_0?????????`?

1?.
result_grads_1?????????`?

? "-?*

 
$?!
1?????????`?
?
"__inference_internal_grad_fn_48021??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_48036??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_48051??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_48066??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_48081??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_48096??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_48111??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_48126??w?t
m?j

 
1?.
result_grads_0?????????`?

1?.
result_grads_1?????????`?

? "-?*

 
$?!
1?????????`?
?
"__inference_internal_grad_fn_48141??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_48156??w?t
m?j

 
1?.
result_grads_0?????????? 
1?.
result_grads_1?????????? 
? "-?*

 
$?!
1?????????? ?
"__inference_internal_grad_fn_48171??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_48186??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_48201??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_48216??u?r
k?h

 
0?-
result_grads_0?????????`
0?-
result_grads_1?????????`
? ",?)

 
#? 
1?????????`?
"__inference_internal_grad_fn_48231??u?r
k?h

 
0?-
result_grads_0?????????
0?-
result_grads_1?????????
? ",?)

 
#? 
1??????????
"__inference_internal_grad_fn_48246??w?t
m?j

 
1?.
result_grads_0?????????`?

1?.
result_grads_1?????????`?

? "-?*

 
$?!
1?????????`?
?
B__inference_post_bn_layer_call_and_return_conditional_losses_47226?????N?K
D?A
;?8
inputs,????????????????????????????

p 
? "@?=
6?3
0,????????????????????????????

? ?
B__inference_post_bn_layer_call_and_return_conditional_losses_47244?????N?K
D?A
;?8
inputs,????????????????????????????

p
? "@?=
6?3
0,????????????????????????????

? ?
B__inference_post_bn_layer_call_and_return_conditional_losses_47262x????<?9
2?/
)?&
inputs?????????`?

p 
? ".?+
$?!
0?????????`?

? ?
B__inference_post_bn_layer_call_and_return_conditional_losses_47280x????<?9
2?/
)?&
inputs?????????`?

p
? ".?+
$?!
0?????????`?

? ?
'__inference_post_bn_layer_call_fn_47169?????N?K
D?A
;?8
inputs,????????????????????????????

p 
? "3?0,????????????????????????????
?
'__inference_post_bn_layer_call_fn_47182?????N?K
D?A
;?8
inputs,????????????????????????????

p
? "3?0,????????????????????????????
?
'__inference_post_bn_layer_call_fn_47195k????<?9
2?/
)?&
inputs?????????`?

p 
? "!??????????`?
?
'__inference_post_bn_layer_call_fn_47208k????<?9
2?/
)?&
inputs?????????`?

p
? "!??????????`?
?
D__inference_post_conv_layer_call_and_return_conditional_losses_47156m?7?4
-?*
(?%
inputs?????????`
? ".?+
$?!
0?????????`?

? ?
)__inference_post_conv_layer_call_fn_47149`?7?4
-?*
(?%
inputs?????????`
? "!??????????`?
?
E__inference_post_swish_layer_call_and_return_conditional_losses_47295j8?5
.?+
)?&
inputs?????????`?

? ".?+
$?!
0?????????`?

? ?
*__inference_post_swish_layer_call_fn_47285]8?5
.?+
)?&
inputs?????????`?

? "!??????????`?
?
F__inference_predictions_layer_call_and_return_conditional_losses_47364_??0?-
&?#
!?
inputs??????????

? "%?"
?
0?????????
? ?
+__inference_predictions_layer_call_fn_47353R??0?-
&?#
!?
inputs??????????

? "???????????
#__inference_signature_wrapper_45064?Y6<=>?HNOPQ[\efsyz{|???????????????????????????????????D?A
? 
:?7
5
input_1*?'
input_1????????? ?"9?6
4
predictions%?"
predictions??????????
P__inference_stack_0_block0_MB_dw__layer_call_and_return_conditional_losses_45989mH8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
5__inference_stack_0_block0_MB_dw__layer_call_fn_45980`H8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46059?NOPQM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46077?NOPQM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46095tNOPQ<?9
2?/
)?&
inputs?????????? 
p 
? ".?+
$?!
0?????????? 
? ?
R__inference_stack_0_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46113tNOPQ<?9
2?/
)?&
inputs?????????? 
p
? ".?+
$?!
0?????????? 
? ?
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46002?NOPQM?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46015?NOPQM?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46028gNOPQ<?9
2?/
)?&
inputs?????????? 
p 
? "!??????????? ?
7__inference_stack_0_block0_MB_dw_bn_layer_call_fn_46041gNOPQ<?9
2?/
)?&
inputs?????????? 
p
? "!??????????? ?
U__inference_stack_0_block0_MB_dw_swish_layer_call_and_return_conditional_losses_46128j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
:__inference_stack_0_block0_MB_dw_swish_layer_call_fn_46118]8?5
.?+
)?&
inputs?????????? 
? "!??????????? ?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46287?yz{|M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46305?yz{|M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46323tyz{|<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
R__inference_stack_0_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46341tyz{|<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46230?yz{|M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46243?yz{|M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46256gyz{|<?9
2?/
)?&
inputs??????????
p 
? "!????????????
7__inference_stack_0_block0_MB_pw_bn_layer_call_fn_46269gyz{|<?9
2?/
)?&
inputs??????????
p
? "!????????????
T__inference_stack_0_block0_MB_pw_conv_layer_call_and_return_conditional_losses_46217ms8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0??????????
? ?
9__inference_stack_0_block0_MB_pw_conv_layer_call_fn_46210`s8?5
.?+
)?&
inputs?????????? 
? "!????????????
P__inference_stack_0_block0_output_layer_call_and_return_conditional_losses_46350j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
5__inference_stack_0_block0_output_layer_call_fn_46346]8?5
.?+
)?&
inputs??????????
? "!????????????
S__inference_stack_0_block0_se_1_conv_layer_call_and_return_conditional_losses_46147l[\7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????
? ?
8__inference_stack_0_block0_se_1_conv_layer_call_fn_46137_[\7?4
-?*
(?%
inputs????????? 
? " ???????????
S__inference_stack_0_block0_se_2_conv_layer_call_and_return_conditional_losses_46181lef7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
8__inference_stack_0_block0_se_2_conv_layer_call_fn_46171_ef7?4
-?*
(?%
inputs?????????
? " ?????????? ?
P__inference_stack_0_block0_se_out_layer_call_and_return_conditional_losses_46203?k?h
a?^
\?Y
+?(
inputs/0?????????? 
*?'
inputs/1????????? 
? ".?+
$?!
0?????????? 
? ?
5__inference_stack_0_block0_se_out_layer_call_fn_46197?k?h
a?^
\?Y
+?(
inputs/0?????????? 
*?'
inputs/1????????? 
? "!??????????? ?
T__inference_stack_0_block0_se_sigmoid_layer_call_and_return_conditional_losses_46191h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
9__inference_stack_0_block0_se_sigmoid_layer_call_fn_46186[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
R__inference_stack_0_block0_se_swish_layer_call_and_return_conditional_losses_46162h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
7__inference_stack_0_block0_se_swish_layer_call_fn_46152[7?4
-?*
(?%
inputs?????????
? " ???????????
P__inference_stack_1_block0_MB_dw__layer_call_and_return_conditional_losses_46366m?8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????`
? ?
5__inference_stack_1_block0_MB_dw__layer_call_fn_46357`?8?5
.?+
)?&
inputs??????????
? " ??????????`?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46436?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46454?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46472v????;?8
1?.
(?%
inputs?????????`
p 
? "-?*
#? 
0?????????`
? ?
R__inference_stack_1_block0_MB_dw_bn_layer_call_and_return_conditional_losses_46490v????;?8
1?.
(?%
inputs?????????`
p
? "-?*
#? 
0?????????`
? ?
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46379?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46392?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46405i????;?8
1?.
(?%
inputs?????????`
p 
? " ??????????`?
7__inference_stack_1_block0_MB_dw_bn_layer_call_fn_46418i????;?8
1?.
(?%
inputs?????????`
p
? " ??????????`?
U__inference_stack_1_block0_MB_dw_swish_layer_call_and_return_conditional_losses_46505h7?4
-?*
(?%
inputs?????????`
? "-?*
#? 
0?????????`
? ?
:__inference_stack_1_block0_MB_dw_swish_layer_call_fn_46495[7?4
-?*
(?%
inputs?????????`
? " ??????????`?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46664?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46682?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46700v????;?8
1?.
(?%
inputs?????????`
p 
? "-?*
#? 
0?????????`
? ?
R__inference_stack_1_block0_MB_pw_bn_layer_call_and_return_conditional_losses_46718v????;?8
1?.
(?%
inputs?????????`
p
? "-?*
#? 
0?????????`
? ?
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46607?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46620?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46633i????;?8
1?.
(?%
inputs?????????`
p 
? " ??????????`?
7__inference_stack_1_block0_MB_pw_bn_layer_call_fn_46646i????;?8
1?.
(?%
inputs?????????`
p
? " ??????????`?
T__inference_stack_1_block0_MB_pw_conv_layer_call_and_return_conditional_losses_46594l?7?4
-?*
(?%
inputs?????????`
? "-?*
#? 
0?????????`
? ?
9__inference_stack_1_block0_MB_pw_conv_layer_call_fn_46587_?7?4
-?*
(?%
inputs?????????`
? " ??????????`?
P__inference_stack_1_block0_output_layer_call_and_return_conditional_losses_46727h7?4
-?*
(?%
inputs?????????`
? "-?*
#? 
0?????????`
? ?
5__inference_stack_1_block0_output_layer_call_fn_46723[7?4
-?*
(?%
inputs?????????`
? " ??????????`?
S__inference_stack_1_block0_se_1_conv_layer_call_and_return_conditional_losses_46524n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
8__inference_stack_1_block0_se_1_conv_layer_call_fn_46514a??7?4
-?*
(?%
inputs?????????
? " ???????????
S__inference_stack_1_block0_se_2_conv_layer_call_and_return_conditional_losses_46558n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
8__inference_stack_1_block0_se_2_conv_layer_call_fn_46548a??7?4
-?*
(?%
inputs?????????
? " ???????????
P__inference_stack_1_block0_se_out_layer_call_and_return_conditional_losses_46580?j?g
`?]
[?X
*?'
inputs/0?????????`
*?'
inputs/1?????????
? "-?*
#? 
0?????????`
? ?
5__inference_stack_1_block0_se_out_layer_call_fn_46574?j?g
`?]
[?X
*?'
inputs/0?????????`
*?'
inputs/1?????????
? " ??????????`?
T__inference_stack_1_block0_se_sigmoid_layer_call_and_return_conditional_losses_46568h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
9__inference_stack_1_block0_se_sigmoid_layer_call_fn_46563[7?4
-?*
(?%
inputs?????????
? " ???????????
R__inference_stack_1_block0_se_swish_layer_call_and_return_conditional_losses_46539h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
7__inference_stack_1_block0_se_swish_layer_call_fn_46529[7?4
-?*
(?%
inputs?????????
? " ???????????
P__inference_stack_1_block1_MB_dw__layer_call_and_return_conditional_losses_46743l?7?4
-?*
(?%
inputs?????????`
? "-?*
#? 
0?????????`
? ?
5__inference_stack_1_block1_MB_dw__layer_call_fn_46734_?7?4
-?*
(?%
inputs?????????`
? " ??????????`?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46813?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46831?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46849v????;?8
1?.
(?%
inputs?????????`
p 
? "-?*
#? 
0?????????`
? ?
R__inference_stack_1_block1_MB_dw_bn_layer_call_and_return_conditional_losses_46867v????;?8
1?.
(?%
inputs?????????`
p
? "-?*
#? 
0?????????`
? ?
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46756?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46769?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46782i????;?8
1?.
(?%
inputs?????????`
p 
? " ??????????`?
7__inference_stack_1_block1_MB_dw_bn_layer_call_fn_46795i????;?8
1?.
(?%
inputs?????????`
p
? " ??????????`?
U__inference_stack_1_block1_MB_dw_swish_layer_call_and_return_conditional_losses_46882h7?4
-?*
(?%
inputs?????????`
? "-?*
#? 
0?????????`
? ?
:__inference_stack_1_block1_MB_dw_swish_layer_call_fn_46872[7?4
-?*
(?%
inputs?????????`
? " ??????????`?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47041?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47059?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47077v????;?8
1?.
(?%
inputs?????????`
p 
? "-?*
#? 
0?????????`
? ?
R__inference_stack_1_block1_MB_pw_bn_layer_call_and_return_conditional_losses_47095v????;?8
1?.
(?%
inputs?????????`
p
? "-?*
#? 
0?????????`
? ?
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_46984?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_46997?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_47010i????;?8
1?.
(?%
inputs?????????`
p 
? " ??????????`?
7__inference_stack_1_block1_MB_pw_bn_layer_call_fn_47023i????;?8
1?.
(?%
inputs?????????`
p
? " ??????????`?
T__inference_stack_1_block1_MB_pw_conv_layer_call_and_return_conditional_losses_46971l?7?4
-?*
(?%
inputs?????????`
? "-?*
#? 
0?????????`
? ?
9__inference_stack_1_block1_MB_pw_conv_layer_call_fn_46964_?7?4
-?*
(?%
inputs?????????`
? " ??????????`?
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_47110l;?8
1?.
(?%
inputs?????????`
p 
? "-?*
#? 
0?????????`
? ?
R__inference_stack_1_block1_dropdrop_layer_call_and_return_conditional_losses_47130l;?8
1?.
(?%
inputs?????????`
p
? "-?*
#? 
0?????????`
? ?
7__inference_stack_1_block1_dropdrop_layer_call_fn_47100_;?8
1?.
(?%
inputs?????????`
p 
? " ??????????`?
7__inference_stack_1_block1_dropdrop_layer_call_fn_47105_;?8
1?.
(?%
inputs?????????`
p
? " ??????????`?
P__inference_stack_1_block1_output_layer_call_and_return_conditional_losses_47142?j?g
`?]
[?X
*?'
inputs/0?????????`
*?'
inputs/1?????????`
? "-?*
#? 
0?????????`
? ?
5__inference_stack_1_block1_output_layer_call_fn_47136?j?g
`?]
[?X
*?'
inputs/0?????????`
*?'
inputs/1?????????`
? " ??????????`?
S__inference_stack_1_block1_se_1_conv_layer_call_and_return_conditional_losses_46901n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
8__inference_stack_1_block1_se_1_conv_layer_call_fn_46891a??7?4
-?*
(?%
inputs?????????
? " ???????????
S__inference_stack_1_block1_se_2_conv_layer_call_and_return_conditional_losses_46935n??7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
8__inference_stack_1_block1_se_2_conv_layer_call_fn_46925a??7?4
-?*
(?%
inputs?????????
? " ???????????
P__inference_stack_1_block1_se_out_layer_call_and_return_conditional_losses_46957?j?g
`?]
[?X
*?'
inputs/0?????????`
*?'
inputs/1?????????
? "-?*
#? 
0?????????`
? ?
5__inference_stack_1_block1_se_out_layer_call_fn_46951?j?g
`?]
[?X
*?'
inputs/0?????????`
*?'
inputs/1?????????
? " ??????????`?
T__inference_stack_1_block1_se_sigmoid_layer_call_and_return_conditional_losses_46945h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
9__inference_stack_1_block1_se_sigmoid_layer_call_fn_46940[7?4
-?*
(?%
inputs?????????
? " ???????????
R__inference_stack_1_block1_se_swish_layer_call_and_return_conditional_losses_46916h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
7__inference_stack_1_block1_se_swish_layer_call_fn_46906[7?4
-?*
(?%
inputs?????????
? " ???????????
B__inference_stem_bn_layer_call_and_return_conditional_losses_45904?<=>?M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
B__inference_stem_bn_layer_call_and_return_conditional_losses_45922?<=>?M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
B__inference_stem_bn_layer_call_and_return_conditional_losses_45940t<=>?<?9
2?/
)?&
inputs?????????? 
p 
? ".?+
$?!
0?????????? 
? ?
B__inference_stem_bn_layer_call_and_return_conditional_losses_45958t<=>?<?9
2?/
)?&
inputs?????????? 
p
? ".?+
$?!
0?????????? 
? ?
'__inference_stem_bn_layer_call_fn_45847?<=>?M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
'__inference_stem_bn_layer_call_fn_45860?<=>?M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
'__inference_stem_bn_layer_call_fn_45873g<=>?<?9
2?/
)?&
inputs?????????? 
p 
? "!??????????? ?
'__inference_stem_bn_layer_call_fn_45886g<=>?<?9
2?/
)?&
inputs?????????? 
p
? "!??????????? ?
D__inference_stem_conv_layer_call_and_return_conditional_losses_45834m68?5
.?+
)?&
inputs????????? ?
? ".?+
$?!
0?????????? 
? ?
)__inference_stem_conv_layer_call_fn_45827`68?5
.?+
)?&
inputs????????? ?
? "!??????????? ?
E__inference_stem_swish_layer_call_and_return_conditional_losses_45973j8?5
.?+
)?&
inputs?????????? 
? ".?+
$?!
0?????????? 
? ?
*__inference_stem_swish_layer_call_fn_45963]8?5
.?+
)?&
inputs?????????? 
? "!??????????? 