#!/usr/bin/env python3
"""Benchmark: batched GEMM vs for-loop, eager + CUDA graph."""
import sys, torch
sys.path.insert(0, '/workspace/fp4_matmul/ThunderKittens/kernels/gemm/nvfp4_b200')
from _C import nvfp4_quantize, nvfp4_batched_gemm, nvfp4_gemm

def quantize(A):
    M,Kh=A.shape; K=Kh*2; fp4=torch.empty(M,Kh,dtype=torch.float4_e2m1fn_x2,device='cuda'); sc=torch.empty(M//128,K//64,512,dtype=torch.float8_e4m3fn,device='cuda'); sg=torch.empty(1,dtype=torch.float32,device='cuda'); nvfp4_quantize(A,fp4,sc,sg,False); return fp4,sc,sg

shapes = [
    ('Attn proj', 16384, 2048, 2048),
    ('QKV dgrad', 16384, 2048, 6144),
    ('FFN gate',  16384, 5632, 2048),
]

def gpu_time_us(fn, iters=30):
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    for _ in range(5): fn()
    torch.cuda.synchronize(); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters*1000

for name, M, K, N in shapes:
    print(f'=== {name}: M={M} K={K} N={N} ===')
    Aq,Bq=[],[]
    for b in range(5):
        Aq.append(quantize(torch.randn(M,K,dtype=torch.bfloat16,device='cuda')))
        Bq.append(quantize(torch.randn(N,K,dtype=torch.bfloat16,device='cuda')))
    torch.cuda.synchronize()
    Ds=[torch.zeros(M,N,dtype=torch.bfloat16,device='cuda') for _ in range(5)]

    for nb in [1,3,5]:
        t_std=gpu_time_us(lambda: [nvfp4_gemm(Aq[i][0],Aq[i][1],Aq[i][2],Bq[i][0],Bq[i][1],Bq[i][2],Ds[i]) for i in range(nb)])
        D=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
        t_bat=gpu_time_us(lambda: nvfp4_batched_gemm(
            [Aq[i][0] for i in range(nb)],[Aq[i][1] for i in range(nb)],[Aq[i][2] for i in range(nb)],
            [Bq[i][0] for i in range(nb)],[Bq[i][1] for i in range(nb)],[Bq[i][2] for i in range(nb)],D))
        print(f'  x{nb}: for-loop={t_std:.0f}us ({t_std/nb:.0f}/g)  batched={t_bat:.0f}us ({t_bat/nb:.0f}/g)  ratio={t_bat/t_std:.2f}x')
    D_ref=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
    nvfp4_gemm(Aq[0][0],Aq[0][1],Aq[0][2],Bq[0][0],Bq[0][1],Bq[0][2],D_ref); torch.cuda.synchronize()
    D_bat=torch.zeros(M,N,dtype=torch.bfloat16,device='cuda')
    nvfp4_batched_gemm([Aq[0][0]],[Aq[0][1]],[Aq[0][2]],[Bq[0][0]],[Bq[0][1]],[Bq[0][2]],D_bat); torch.cuda.synchronize()
    nr=D_ref.isnan().any().item(); nb2=D_bat.isnan().any().item()
    if not nr and not nb2:
        print(f'  Correctness: exact={(D_ref==D_bat).all().item()} maxdiff={(D_ref.float()-D_bat.float()).abs().max().item():.6f}')
    else: print(f'  NaN: ref={nr} bat={nb2}')
    print()

# CUDA graph
print('=== CUDA Graph: Attn proj x5 ===')
nb2=5
g1=torch.cuda.CUDAGraph(); g2=torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    for i in range(nb2): nvfp4_gemm(Aq[i][0],Aq[i][1],Aq[i][2],Bq[i][0],Bq[i][1],Bq[i][2],Ds[i])
D3=torch.zeros(16384,2048,dtype=torch.bfloat16,device='cuda')
with torch.cuda.graph(g2):
    nvfp4_batched_gemm([Aq[i][0] for i in range(nb2)],[Aq[i][1] for i in range(nb2)],[Aq[i][2] for i in range(nb2)],
        [Bq[i][0] for i in range(nb2)],[Bq[i][1] for i in range(nb2)],[Bq[i][2] for i in range(nb2)],D3)
t1=gpu_time_us(lambda:g1.replay()); t2=gpu_time_us(lambda:g2.replay())
print(f'  graph for-loop={t1:.0f}us ({t1/nb2:.0f}/g)  graph batched={t2:.0f}us ({t2/nb2:.0f}/g)  ratio={t2/t1:.2f}x')
print('Done')
