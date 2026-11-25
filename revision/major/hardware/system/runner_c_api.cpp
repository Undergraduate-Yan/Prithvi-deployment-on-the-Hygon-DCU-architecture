#include "k100_pipeline_runner.hpp"
#include <exception>
#include <string>
static thread_local std::string error;
extern "C" {
const char* rs_error(){return error.c_str();}
void* rs_create(const char* plan,const char* profile){
 try{return new k100::PipelineRunner(k100::LoadPipelinePlan(plan),0,profile);}
 catch(const std::exception& e){error=e.what();return nullptr;}
}
int rs_run(void* ptr,const float* input,float* output){
 try{auto* r=static_cast<k100::PipelineRunner*>(ptr);r->CopyInputFromHost(input,r->input_elements());r->RunPipeline();r->CopyLogitsToHost(output,r->logits_elements());return 0;}
 catch(const std::exception& e){error=e.what();return 1;}
}
void rs_destroy(void* ptr){delete static_cast<k100::PipelineRunner*>(ptr);}
}
