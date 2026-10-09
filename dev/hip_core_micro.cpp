// Development-only FP64 batched Jacobian-core measurement. No production
// solver, allocator or Julia dependency is changed. Inputs are exported by
// hip_core_input.jl; all 105 output matrices are checked against its CPU result.
// API: https://rocm.docs.amd.com/projects/hipBLAS/en/latest/reference/hipblas-api-functions.html
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void hip_ok(hipError_t s) { if(s!=hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void blas_ok(hipblasStatus_t s) { if(s!=HIPBLAS_STATUS_SUCCESS) throw std::runtime_error("hipBLAS status "+std::to_string(int(s))); }
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(size_t n) { void* v=nullptr; hip_ok(hipMalloc(&v,n*sizeof(T))); p=static_cast<T*>(v); }
    ~Device() { if(p) (void)hipFree(p); }
    Device(const Device&)=delete; Device& operator=(const Device&)=delete;
};
struct Blas {
    hipblasHandle_t h=nullptr;
    Blas() { blas_ok(hipblasCreate(&h)); }
    ~Blas() { if(h) (void)hipblasDestroy(h); }
};
template<class T> void read_exact(std::ifstream& in,T* p,size_t n) {
    if(!in.read(reinterpret_cast<char*>(p),std::streamsize(n*sizeof(T)))) throw std::runtime_error("truncated core input");
}
__global__ void symmetrize(double* C,int d,int pairs) {
    size_t id=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=size_t(d)*d*pairs) return;
    int i=int(id%size_t(d)); int j=int((id/size_t(d))%size_t(d));
    if(i>j) return;
    size_t base=(id/(size_t(d)*d))*size_t(d)*d;
    double v=(C[base+i+size_t(j)*d]+C[base+j+size_t(i)*d])/2.0;
    C[base+i+size_t(j)*d]=v; C[base+j+size_t(i)*d]=v;
}
template<class F> double elapsed(F f) {
    auto t=std::chrono::steady_clock::now(); f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
}
template<class F> void report(const char* name,F f) {
    std::vector<double> times;
    for(int i=0;i<7;++i) times.push_back(elapsed(f));
    std::sort(times.begin(),times.end());
    std::cout<<name<<" median_seconds="<<times[times.size()/2]<<" min="<<times.front()<<" max="<<times.back()<<std::endl;
}
int main(int argc,char** argv) try {
    if(argc!=2) throw std::runtime_error("usage: hip_core_micro <authenticated core input>");
    std::ifstream in(argv[1],std::ios::binary);
    char magic[8]; read_exact(in,magic,8);
    if(std::string(magic,8)!="KTRJC001") throw std::runtime_error("wrong core format");
    uint64_t dims[4]; read_exact(in,dims,4);
    if(dims[0]<1 || dims[0]>128 || dims[1]<1 || dims[1]>2048 || dims[2]<1 || dims[2]>32 || dims[3]!=dims[2]*(dims[2]+1)/2)
        throw std::runtime_error("core dimensions exceed bounded probe");
    int d=int(dims[0]), k=int(dims[1]), nc=int(dims[2]), pairs=int(dims[3]);
    size_t nb=size_t(d)*k*nc, no=size_t(d)*d*pairs;
    std::vector<double> B(nb), A(nb), truth(no), result(no);
    read_exact(in,B.data(),nb); read_exact(in,A.data(),nb); read_exact(in,truth.data(),no);
    char tail; if(in.read(&tail,1)) throw std::runtime_error("extra input bytes");
    for(double x:B) if(!std::isfinite(x)) throw std::runtime_error("nonfinite B");
    for(double x:A) if(!std::isfinite(x)) throw std::runtime_error("nonfinite A");
    int count=0; hip_ok(hipGetDeviceCount(&count)); if(count<1) throw std::runtime_error("no HIP device");
    hip_ok(hipSetDevice(0)); hipDeviceProp_t prop{}; hip_ok(hipGetDeviceProperties(&prop,0));
    int runtime=0; hip_ok(hipRuntimeGetVersion(&runtime));
    std::cout<<std::setprecision(12)<<"device="<<prop.name<<" arch="<<prop.gcnArchName<<" runtime="<<runtime
        <<" FP64 d="<<d<<" rank="<<k<<" pairs="<<pairs<<std::endl;
    Device<double> da(nb), db(nb), dc(no);
    std::vector<const double*> ap(pairs),bp(pairs); std::vector<double*> cp(pairs);
    int q=0; for(int r=0;r<nc;++r) for(int s=r;s<nc;++s) {
        ap[q]=da.p+size_t(r)*d*k; bp[q]=db.p+size_t(s)*d*k; cp[q]=dc.p+size_t(q)*d*d; ++q;
    }
    Device<const double*> dap(pairs),dbp(pairs); Device<double*> dcp(pairs);
    hip_ok(hipMemcpy(dap.p,ap.data(),pairs*sizeof(double*),hipMemcpyHostToDevice));
    hip_ok(hipMemcpy(dbp.p,bp.data(),pairs*sizeof(double*),hipMemcpyHostToDevice));
    hip_ok(hipMemcpy(dcp.p,cp.data(),pairs*sizeof(double*),hipMemcpyHostToDevice));
    Blas blas; const double one=1.0, zero=0.0;
    auto upload=[&](){hip_ok(hipMemcpy(da.p,A.data(),nb*sizeof(double),hipMemcpyHostToDevice)); hip_ok(hipMemcpy(db.p,B.data(),nb*sizeof(double),hipMemcpyHostToDevice));};
    auto compute=[&](){
        blas_ok(hipblasDgemmBatched(blas.h,HIPBLAS_OP_N,HIPBLAS_OP_T,d,d,k,&one,dap.p,d,dbp.p,d,&zero,dcp.p,d,pairs));
        hipLaunchKernelGGL(symmetrize,dim3(unsigned((no+255)/256)),dim3(256),0,0,dc.p,d,pairs);
        hip_ok(hipGetLastError());
    };
    upload();
    std::cout<<"first_compute_seconds="<<elapsed([&](){compute(); hip_ok(hipDeviceSynchronize());})<<std::endl;
    hip_ok(hipMemcpy(result.data(),dc.p,no*sizeof(double),hipMemcpyDeviceToHost));
    double max_abs=0.0; long double err2=0.0, ref2=0.0; size_t bad=0;
    for(size_t i=0;i<no;++i) {
        double delta=std::abs(result[i]-truth[i]);
        if(!std::isfinite(result[i]) || delta>1e-12+1e-12*std::abs(truth[i])) ++bad;
        max_abs=std::max(max_abs,delta); err2+=static_cast<long double>(delta)*delta; ref2+=static_cast<long double>(truth[i])*truth[i];
    }
    std::cout<<"elements="<<no<<" mismatch_count="<<bad<<" max_abs="<<max_abs<<" relative_frobenius="<<std::sqrt(err2/std::max(ref2,1e-300L))<<std::endl;
    if(bad) throw std::runtime_error("original per-element core tolerance failed");
    report("resident_compute_and_sync",[&](){compute();hip_ok(hipDeviceSynchronize());});
    report("alpha_upload_compute_download",[&](){hip_ok(hipMemcpy(da.p,A.data(),nb*sizeof(double),hipMemcpyHostToDevice));compute();hip_ok(hipMemcpy(result.data(),dc.p,no*sizeof(double),hipMemcpyDeviceToHost));});
    report("all_upload_compute_download",[&](){upload();compute();hip_ok(hipMemcpy(result.data(),dc.p,no*sizeof(double),hipMemcpyDeviceToHost));});
    std::cout<<"input_bytes="<<2*nb*sizeof(double)<<" output_bytes="<<no*sizeof(double)
        <<"; no EB/certificates/backtest; no whole-solver acceleration claim"<<std::endl;
    return 0;
} catch(const std::exception& e) { std::cerr<<"ERROR "<<e.what()<<std::endl; return 1; }
