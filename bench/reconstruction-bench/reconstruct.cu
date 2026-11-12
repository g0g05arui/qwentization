// Diagnostic decoders, NOT the optimized llama.cpp matvec kernels.
// Layouts are validated at run time against this installation's ggml CPU decoder.
#include "ggml.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CU(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
 fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(2); }} while(0)

// All three use the same 256-thread geometry and FP32 reduction.
constexpr int THREADS=256;
enum Format { Q40, Q4K, IQ4XS };
enum Variant { FULL, NO_META, NO_MAP, CODES_ONLY };
enum Mode { WRITE_HALF, CHECKSUM, MATVEC, READ_BYTES };
static const char * formats[]={"q4_0","q4_K","iq4_xs"};
static const char * variants[]={"full","no_metadata_DIAGNOSTIC","no_mapping_DIAGNOSTIC","codes_only_DIAGNOSTIC"};
static const char * modes[]={"write_fp16","checksum","matvec","read_bytes_control"};
static const ggml_type types[]={GGML_TYPE_Q4_0,GGML_TYPE_Q4_K,GGML_TYPE_IQ4_XS};

// GGUF block layouts: half values are little-endian 16-bit bit patterns.
struct Q40Block { uint16_t d; uint8_t qs[16]; };
struct Q4KBlock { uint16_t d,dmin; uint8_t scales[12],qs[128]; };
struct IQ4XSBlock { uint16_t d,sh; uint8_t sl[4],qs[128]; };
static_assert(sizeof(Q40Block)==18 && sizeof(Q4KBlock)==144 && sizeof(IQ4XSBlock)==136);

// Runtime initialized: the compiler cannot assume a constant mapping table.
// This is a simple lookup implementation; llama.cpp may implement it with PRMT.
__device__ __constant__ int iq_table[16];
static const int host_table[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

template<int F,int V> __device__ __forceinline__ float decode(const uint8_t * raw,size_t i) {
    constexpr bool meta=(V!=NO_META && V!=CODES_ONLY);
    constexpr bool mapping=(V!=NO_MAP && V!=CODES_ONLY);
    if constexpr(F==Q40) {
        const auto & b=reinterpret_cast<const Q40Block *>(raw)[i/32];
        const int t=i%32;
        int code=(b.qs[t%16] >> (4*(t/16)))&15;
        float v=mapping ? float(code-8) : float(code);
        return meta ? __half2float(__ushort_as_half(b.d))*v : v;
    } else if constexpr(F==Q4K) {
        const auto & b=reinterpret_cast<const Q4KBlock *>(raw)[i/256];
        const int t=i%256, sb=t/32;
        const int code=(b.qs[(t/64)*32+t%32] >> (4*((t/32)%2)))&15;
        if constexpr(meta) {
            int sc,mn;
            if(sb<4) { sc=b.scales[sb]&63; mn=b.scales[sb+4]&63; }
            else { sc=(b.scales[sb+4]&15)|((b.scales[sb-4]>>6)<<4);
                   mn=(b.scales[sb+4]>>4)|((b.scales[sb]>>6)<<4); }
            const float d=__half2float(__ushort_as_half(b.d));
            const float dm=__half2float(__ushort_as_half(b.dmin));
            return (d*sc)*code-dm*mn;
        } else return float(code);
    } else {
        const auto & b=reinterpret_cast<const IQ4XSBlock *>(raw)[i/256];
        const int t=i%256,sb=t/32;
        const int code=(b.qs[sb*16+t%16] >> (4*((t/16)%2)))&15;
        const float v=mapping ? float(iq_table[code]) : float(code-8);
        if constexpr(meta) {
            const int sc=((b.sl[sb/2]>>(4*(sb%2)))&15)|(((b.sh>>(2*sb))&3)<<4);
            return (__half2float(__ushort_as_half(b.d))*(sc-32))*v;
        } else return v;
    }
}

__device__ float reduce(float v) {
    __shared__ float partial[8];
    for(int d=16;d;d/=2) v+=__shfl_down_sync(0xffffffff,v,d);
    if((threadIdx.x&31)==0) partial[threadIdx.x/32]=v;
    __syncthreads();
    v=threadIdx.x<8 ? partial[threadIdx.x] : 0;
    if(threadIdx.x<32) for(int d=16;d;d/=2) v+=__shfl_down_sync(0xffffffff,v,d);
    return v;
}

template<int F,int V,int M> __global__ void diagnostic_kernel(
    const uint8_t * raw,const float * x,__half * out,float * sums,
    size_t n,size_t nbytes,int k) {
    float acc=0;
    if constexpr(M==MATVEC) {
        const size_t row=blockIdx.x;
        for(int col=threadIdx.x;col<k;col+=blockDim.x)
            acc=fmaf(decode<F,V>(raw,row*k+col),x[col],acc);
    } else if constexpr(M==READ_BYTES) {
        // Reads all packed bytes, but not the same memory instructions as decode.
        // This is a traffic control, NOT a subtractable "memory-only time".
        for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<nbytes;i+=size_t(gridDim.x)*blockDim.x)
            acc+=raw[i];
    } else {
        for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
            float v=decode<F,V>(raw,i);
            if constexpr(M==WRITE_HALF) out[i]=__float2half_rn(v);
            else acc+=v;
        }
    }
    if constexpr(M!=WRITE_HALF) {
        acc=reduce(acc);
        if(threadIdx.x==0) sums[blockIdx.x]=acc;
    }
}

struct Options {
    int m=17408,k=5120,nmat=4,repeat=3,seed=471;
    double seconds=1;
    bool profile=false,validate_only=false;
    std::string format="all",out="reconstruction.csv";
};
struct Data {
    Options o;
    size_t n=0,nbytes=0,rows=0;
    int grid=0;
    uint8_t * raw=nullptr;
    float * x=nullptr,* sums=nullptr;
    __half * output=nullptr;
    cudaStream_t stream=nullptr;
    std::vector<uint8_t> host;
    std::vector<float> hx;
    ~Data() { cudaFree(raw);cudaFree(x);cudaFree(sums);cudaFree(output);cudaStreamDestroy(stream); }
};

static void make_data(Data & d,int f) {
    d.rows=size_t(d.o.m)*d.o.nmat; d.n=d.rows*d.o.k;
    const size_t rs=ggml_row_size(types[f],d.o.k);
    d.nbytes=rs*d.rows;
    if(rs!=size_t(d.o.k/(f==Q40?32:256))*(f==Q40?18:f==Q4K?144:136))
        throw std::runtime_error("ggml block layout mismatch");
    d.host.resize(d.nbytes); d.hx.resize(d.o.k);
    std::mt19937 xgen(d.o.seed); std::uniform_real_distribution<float> dist(-1,1);
    for(auto & x:d.hx) x=dist(xgen);
    int nt=std::min<size_t>(16,d.rows);
    std::vector<std::thread> workers;
    for(int t=0;t<nt;t++) workers.emplace_back([&,t] {
        std::vector<float> row(d.o.k),importance(d.o.k,1.f);
        // Seed per row: same original values for every format and thread count.
        for(size_t r=t;r<d.rows;r+=nt) {
            std::mt19937 gen(unsigned(d.o.seed)+unsigned(r)*7919u);
            std::uniform_real_distribution<float> rd(-1,1);
            for(auto & a:row) a=rd(gen);
            ggml_quantize_chunk(types[f],row.data(),d.host.data()+r*rs,0,1,d.o.k,
                ggml_quantize_requires_imatrix(types[f])?importance.data():nullptr);
        }
    });
    for(auto & t:workers) t.join();
    CU(cudaStreamCreate(&d.stream));
    cudaDeviceProp prop; CU(cudaGetDeviceProperties(&prop,0));
    // Same grid across formats. Small case fits in L2; not asserted to hit L2.
    d.grid=std::min<size_t>((d.n+THREADS-1)/THREADS,prop.multiProcessorCount*8);
    CU(cudaMalloc((void **)&d.raw,d.nbytes));
    CU(cudaMalloc((void **)&d.x,d.o.k*sizeof(float)));
    CU(cudaMalloc((void **)&d.output,d.n*sizeof(__half)));
    CU(cudaMalloc((void **)&d.sums,std::max<size_t>(d.grid,d.rows)*sizeof(float)));
    CU(cudaMemcpy(d.raw,d.host.data(),d.nbytes,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d.x,d.hx.data(),d.o.k*sizeof(float),cudaMemcpyHostToDevice));
    fprintf(stderr,"%s: %zu weights, %zu packed bytes, %d decode CTAs, %zu matvec CTAs\n",
        formats[f],d.n,d.nbytes,d.grid,d.rows);
}

template<int F,int V,int M> static void launch(Data & d) {
    const int grid=M==MATVEC?int(d.rows):d.grid;
    diagnostic_kernel<F,V,M><<<grid,THREADS,0,d.stream>>>(d.raw,d.x,d.output,d.sums,d.n,d.nbytes,d.o.k);
    CU(cudaGetLastError());
}

template<int F> static void validate(Data & d) {
    // Two full launches, sample rows at beginning and end against an independent
    // ggml CPU implementation. These launches are separate from timed samples.
    launch<F,FULL,WRITE_HALF>(d); launch<F,FULL,MATVEC>(d);
    CU(cudaStreamSynchronize(d.stream));
    auto * traits=ggml_get_type_traits(types[F]);
    if(!traits || !traits->to_float) throw std::runtime_error("Missing CPU dequantizer");
    std::vector<float> ref(d.o.k); std::vector<__half> got(d.o.k);
    const size_t rs=ggml_row_size(types[F],d.o.k);
    for(size_t r=0;r<d.rows;r++) {
        if(r>=8 && r+8<d.rows && r!=d.rows/2) continue;
        traits->to_float(d.host.data()+r*rs,ref.data(),d.o.k);
        CU(cudaMemcpy(got.data(),d.output+r*d.o.k,d.o.k*sizeof(__half),cudaMemcpyDeviceToHost));
        double sum=0,absum=0;
        for(int c=0;c<d.o.k;c++) {
            const float expected=__half2float(__float2half_rn(ref[c]));
            // FMA vs separate rounding may differ at FP16 rounding boundaries.
            if(std::abs(__half2float(got[c])-expected)>2e-6f+0.001f*std::abs(expected))
                throw std::runtime_error("Decoded weight differs from ggml CPU reference");
            sum+=double(ref[c])*d.hx[c]; absum+=std::abs(double(ref[c])*d.hx[c]);
        }
        float y; CU(cudaMemcpy(&y,d.sums+r,sizeof(float),cudaMemcpyDeviceToHost));
        if(!std::isfinite(y) || std::abs(y-sum)>1e-5*absum+1e-4)
            throw std::runtime_error("Matvec differs from ggml CPU reference");
    }
    fprintf(stderr,"PASS: %s sampled weight decoding and matvec vs ggml CPU\n",formats[F]);
}

template<int F,int V,int M> static void measure(Data & d,std::ofstream & csv) {
    cudaFuncAttributes attr; CU(cudaFuncGetAttributes(&attr,diagnostic_kernel<F,V,M>));
    if(d.o.profile) {
        // One direct launch per case: no CUDA graphs, calibration, or warmups.
        launch<F,V,M>(d); CU(cudaStreamSynchronize(d.stream));
        fprintf(stderr,"PROFILE format=%s variant=%s mode=%s weights=%zu blocks=%zu threads=%d\n",
          formats[F],variants[V],modes[M],d.n,M==MATVEC?d.rows:size_t(d.grid),THREADS);
        return;
    }
    // CUDA graphs amortize host launch costs, particularly in the small case.
    constexpr int batch=32;
    cudaGraph_t graph; cudaGraphExec_t exec;
    CU(cudaStreamBeginCapture(d.stream,cudaStreamCaptureModeGlobal));
    for(int i=0;i<batch;i++) launch<F,V,M>(d);
    CU(cudaStreamEndCapture(d.stream,&graph));
    CU(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int i=0;i<3;i++) CU(cudaGraphLaunch(exec,d.stream));
    CU(cudaStreamSynchronize(d.stream));
    cudaEvent_t start,stop; CU(cudaEventCreate(&start));CU(cudaEventCreate(&stop));
    CU(cudaEventRecord(start,d.stream)); CU(cudaGraphLaunch(exec,d.stream)); CU(cudaEventRecord(stop,d.stream));
    CU(cudaEventSynchronize(stop)); float ms; CU(cudaEventElapsedTime(&ms,start,stop));
    int loops=std::max(1,int(std::ceil(d.o.seconds*1000/std::max(0.001f,ms))));
    for(int rep=0;rep<d.o.repeat;rep++) {
        CU(cudaEventRecord(start,d.stream));
        for(int j=0;j<loops;j++) CU(cudaGraphLaunch(exec,d.stream));
        CU(cudaEventRecord(stop,d.stream));CU(cudaEventSynchronize(stop));
        CU(cudaEventElapsedTime(&ms,start,stop));
        const double us=ms*1000./(double(loops)*batch);
        float sink=0;
        if constexpr(M==WRITE_HALF) { __half h;CU(cudaMemcpy(&h,d.output,sizeof(h),cudaMemcpyDeviceToHost));sink=__half2float(h); }
        else CU(cudaMemcpy(&sink,d.sums,sizeof(sink),cudaMemcpyDeviceToHost));
        if(!std::isfinite(sink)) throw std::runtime_error("Nonfinite output");
        csv<<formats[F]<<','<<variants[V]<<','<<modes[M]<<','<<d.o.m<<','<<d.o.k<<','<<d.o.nmat<<','
           <<d.n<<','<<d.nbytes<<','<<THREADS<<','<<(M==MATVEC?d.rows:size_t(d.grid))<<','
           <<attr.numRegs<<','<<attr.localSizeBytes<<','<<rep<<','<<loops*batch<<','<<us<<','
           <<us*1000/d.n<<','<<sink<<'\n'; csv.flush();
        fprintf(stderr,"%s %-27s %-18s %.3f us, %d regs, %zu local bytes/thread\n",
            formats[F],variants[V],modes[M],us,attr.numRegs,attr.localSizeBytes);
    }
    CU(cudaEventDestroy(start));CU(cudaEventDestroy(stop));
    CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));
}

template<int F> static void run(Data & d,std::ofstream & csv) {
    make_data(d,F);
    // Instrumented runs skip validation to keep one launch per measured case.
    // Run the normal benchmark first; it always validates before timing.
    if(!d.o.profile) validate<F>(d);
    if(d.o.validate_only) return;
    measure<F,FULL,WRITE_HALF>(d,csv);
    measure<F,FULL,READ_BYTES>(d,csv);
    measure<F,FULL,CHECKSUM>(d,csv);
    measure<F,NO_META,CHECKSUM>(d,csv);
    if constexpr(F!=Q4K) measure<F,NO_MAP,CHECKSUM>(d,csv);
    measure<F,CODES_ONLY,CHECKSUM>(d,csv);
    measure<F,FULL,MATVEC>(d,csv);
    measure<F,NO_META,MATVEC>(d,csv);
    if constexpr(F!=Q4K) measure<F,NO_MAP,MATVEC>(d,csv);
    measure<F,CODES_ONLY,MATVEC>(d,csv);
}

int main(int argc,char ** argv) {
  try {
    Options o;
    for(int i=1;i<argc;i++) {
        std::string a=argv[i];
        auto next=[&]() -> std::string { if(++i>=argc) throw std::runtime_error("Missing argument"); return argv[i]; };
        if(a=="--m") o.m=std::stoi(next()); else if(a=="--k") o.k=std::stoi(next());
        else if(a=="--nmat") o.nmat=std::stoi(next()); else if(a=="--seconds") o.seconds=std::stod(next());
        else if(a=="--repeat") o.repeat=std::stoi(next()); else if(a=="--format") o.format=next();
        else if(a=="--out") o.out=next(); else if(a=="--seed") o.seed=std::stoi(next());
        else if(a=="--profile") o.profile=true; else if(a=="--validate-only") o.validate_only=true;
        else throw std::runtime_error("Unknown option: "+a);
    }
    if(o.m<1 || o.k<256 || o.k%256 || o.nmat<1 || o.repeat<1 || !std::isfinite(o.seconds) || o.seconds<=0)
        throw std::runtime_error("Require m,nmat,repeat > 0, k divisible by 256, seconds > 0");
    if(size_t(o.m)*o.nmat>2147483647) throw std::runtime_error("Too many rows");
    if(o.format!="all" && o.format!="q4_0" && o.format!="q4_K" && o.format!="iq4_xs")
        throw std::runtime_error("format: all, q4_0, q4_K, iq4_xs");
    CU(cudaSetDevice(0)); CU(cudaMemcpyToSymbol(iq_table,host_table,sizeof(host_table)));
    ggml_init_params init={1024*1024,nullptr,true};
    ggml_context * ctx=ggml_init(init); if(!ctx) throw std::runtime_error("ggml initialization failed");
    std::ofstream csv(o.out); if(!csv) throw std::runtime_error("Cannot open CSV output");
    csv<<"format,variant,mode,m,k,nmat,weights,input_bytes,threads,blocks,registers,local_bytes,repeat,launches,us_per_launch,ns_per_weight,sink\n";
    // The runner randomizes process order; one format per invocation is preferred.
    for(int f=0;f<3;f++) if(o.format=="all" || o.format==formats[f]) {
        Data d;d.o=o;
        if(f==Q40) run<Q40>(d,csv); else if(f==Q4K) run<Q4K>(d,csv); else run<IQ4XS>(d,csv);
    }
    ggml_free(ctx); return 0;
  } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what());return 1; }
}
