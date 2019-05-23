gpu=0
data=vkitti
model=drn26
name=tmp

crop=None
batch=4
iterations=100000
lr=1e-3
momentum=0.99
num_cls=14

## my param
datadir=/media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/vkitti

#outdir=results/${data}-${data2}/${model}
outdir=results/${data}/${model}/${name}/${data}_${model}
mkdir -p results/${data}/${model}/${name}

python train_fcn_vkitti.py ${outdir} --model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --lr ${lr} -b ${batch} -m ${momentum} \
    --crop_size ${crop} --iterations ${iterations} \
    --datadir ${datadir} \
    --dataset ${data}  #--dataset ${data2} 
