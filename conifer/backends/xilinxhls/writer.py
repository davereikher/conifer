import os
import sys
from shutil import copyfile
import numpy as np

_TOOLS = {
    'vivadohls': 'vivado_hls',
    'vitishls': 'vitis_hls'
}


def get_tool_exe_in_path(tool):
    if tool not in _TOOLS.keys():
        return None

    tool_exe = _TOOLS[tool]

    if os.system('which {} > /dev/null 2>/dev/null'.format(tool_exe)) != 0:
        return None

    return tool_exe


def get_hls():

    tool_exe = None

    if '_tool' in globals():
        tool_exe = get_tool_exe_in_path(_tool)
    else:
        for tool in _TOOLS.keys():
            tool_exe = get_tool_exe_in_path(tool)
            if tool_exe != None:
                break

    return tool_exe

def make_optimized_optimized_bdt_code_replacements(ensemble_dict, cfg):
    nodes_list=[]
    leaves_list=[]
    trees_list=[]
    decision_functions_list=[]
    for ntree, x in enumerate(ensemble_dict['trees']):
        nodes_list.append('\tcase {}:return {};'.format(ntree, len(x[0]['feature'])))
        leaves_list.append('\tcase {}:return {};'.format(ntree, len([f for f in x[0]['feature'] if f==-2])))
        trees_list.append("\tTree<{0}, input_t, score_t, threshold_t> tree_{0}[fn_classes(n_classes)];".format(ntree))
        dec_func_template = "\t\tfor(int j = 0; j < fn_classes(n_classes); j++){{\n\t\t\tscore_t s = tree_{0}[j].decision_function(x);"\
            "\n\t\t\tscore[j] += s;"
        if cfg['OutputTreeScores']:
            dec_func_template += "\n\t\t\ttree_scores[{0} * fn_classes(n_classes) + j] = s;"
        dec_func_template += "\n\t\t}}"
        decision_functions_list.append(dec_func_template.format(ntree))
    nodes_list.append("\tdefault:return {};".format(2**(ensemble_dict['max_depth'] + 1) - 1))
    leaves_list.append("\tdefault:return {};".format(2**(ensemble_dict['max_depth'])))
    
    switch_case_nodes_code = "\nconstexpr int fn_nodes(int tree_idx) {\n\tswitch  (tree_idx) {" + "\n".join(nodes_list) + "\n}\n"
    switch_case_leaves_code = "\nconstexpr int fn_leaves(int tree_idx){\n\tswitch  (tree_idx) {" + "\n".join(leaves_list) + "\n}\n"
    trees_code = "\n" + "\n".join(trees_list) + "\n"
    decision_functions_code = "\n" + "\n".join(decision_functions_list) + "\n"
    #print(switch_case_nodes)
    #print(switch_case_leaves)
    #print(decision_functions_list)
    
#    with open(path, "r") as f:
#        data=f.read()
#
#    data=data.replace("%%SWITCH_CASE_N_NODES%%", switch_case_nodes)
#    data=data.replace("%%SWITCH_CASE_N_LEAVES%%", switch_case_leaves)
#    data=data.replace("%%TREE_LIST%%", tree_list)
#    data=data.replace("%%DECISION_FUNCTION_LIST%%", decision_functions_list)
#        
#    with open(path, "w") as f:
#        f.write(data)
    return switch_case_nodes_code, switch_case_leaves_code, trees_code, decision_functions_code

def make_bdt_code_replacements(cfg):
    #TODO (Dave): fix indentation
    fn_nodes_code = "\nconstexpr int fn_nodes(int max_depth){\n\treturn pow2(max_depth + 1) - 1;\n"
    fn_leaves_code = "\nconstexpr int fn_leaves(int max_depth){\n\treturn pow2(max_depth);\n"
    trees_code = "\nTree<max_depth, input_t, score_t, threshold_t> trees[n_trees][fn_classes(n_classes)];\n"
    decision_functions_code = """
    	for(int i = 0; i < n_trees; i++){
    		Classes:
    		for(int j = 0; j < fn_classes(n_classes); j++){
            score_t s = trees[i][j].decision_function(x);
    			score[j] += s;"""
    if cfg['OutputTreeScores']:
        decision_functions_code +="\ntree_scores[i * fn_classes(n_classes) + j] = s;\n"
    decision_functions_code += "}}"

    return fn_nodes_code, fn_leaves_code, trees_code, decision_functions_code

def write(ensemble_dict, cfg):
    filedir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs('{}/firmware'.format(cfg['OutputDir']))
    os.makedirs('{}/tb_data'.format(cfg['OutputDir']))

    ###################
    # BDT.h
    ###################

    if cfg['Optimized']:
        fn_nodes_code, fn_leaves_code, trees_code, decision_functions_code = \
                make_optimized_optimized_bdt_code_replacements(ensemble_dict, cfg)
    else:
        fn_nodes_code, fn_leaves_code, trees_code, decision_functions_code = \
                make_bdt_code_replacements(cfg)

    f = open(os.path.join(filedir, 'hls-template/firmware/BDT.h'), 'r')
    fout = open('{}/firmware/BDT.h'.format(cfg['OutputDir']), 'w')

    for line in f.readlines():
        if 'hls-fpga-machine-learning insert switch case nodes' in line:
            newline = fn_nodes_code 
        elif 'hls-fpga-machine-learning insert switch case leaves' in line:
            newline = fn_leaves_code
        elif 'hls-fpga-machine-learning insert trees' in line:
            newline = trees_code
        elif 'hls-fpga-machine-learning insert decision functions' in line:
            newline = decision_functions_code
        elif 'hls-fpga-machine-learning insert declarations' in line:
            if cfg['Optimized']:
                newline = ""
            else:
                newline = "\ntemplate<int max_depth, class input_t, class score_t, class threshold_t>\n"\
                    "constexpr int Tree<max_depth, input_t, score_t, threshold_t>::n_nodes;\n\n"\
                    "template<int max_depth, class input_t, class score_t, class threshold_t>\n"\
                    "constexpr int Tree<max_depth, input_t, score_t, threshold_t>::n_leaves;\n"
        elif 'TREES_PARAM' in line:
            if cfg['OutputTreeScores']:
                newline = line.replace(
                    'TREES_PARAM', ', score_t tree_scores[fn_classes(n_classes) * n_trees]')
            else:
                newline = line.replace('TREES_PARAM', '')
        else:
            newline = line
        fout.write(newline)

    f.close()
    fout.close()

    ###################
    # myproject.cpp
    ###################

    fout = open(
        '{}/firmware/{}.cpp'.format(cfg['OutputDir'], cfg['ProjectName']), 'w')
    fout.write('#include "BDT.h"\n')
    fout.write('#include "parameters.h"\n')
    fout.write('#include "{}.h"\n'.format(cfg['ProjectName']))

    if(cfg['OutputTreeScores']):
        fout.write(
        'void {}(input_arr_t x, score_arr_t score, score_t tree_scores[BDT::fn_classes(n_classes) * n_trees]){{\n'.format(cfg['ProjectName']))
    else:
        fout.write(
            'void {}(input_arr_t x, score_arr_t score){{\n'.format(cfg['ProjectName']))
    fout.write('\t#pragma HLS array_partition variable=x\n')
    fout.write('\t#pragma HLS array_partition variable=score\n')
    if(cfg['OutputTreeScores']):
        fout.write('\t#pragma HLS array_partition variable=tree_scores\n')
    if(not cfg['ControlPort']):
        fout.write('\t#pragma HLS interface ap_ctrl_none port=return\n')
    if(cfg['Pipeline']):
        fout.write('\t#pragma HLS pipeline\n')
        fout.write('\t#pragma HLS unroll\n')
    if(cfg['OutputTreeScores']):
        fout.write('\tbdt.decision_function(x, score, tree_scores);\n}')
    else:
        fout.write('\tbdt.decision_function(x, score);\n}')
    fout.close()

    ###################
    # parameters.h
    ###################

    fout = open('{}/firmware/parameters.h'.format(cfg['OutputDir']), 'w')
    fout.write('#ifndef BDT_PARAMS_H__\n#define BDT_PARAMS_H__\n\n')
    fout.write('#include  "BDT.h"\n')
    fout.write('#include "ap_fixed.h"\n\n')
    fout.write('static const int n_trees = {};\n'.format(
        ensemble_dict['n_trees']))
    fout.write('static const int max_depth = {};\n'.format(
        ensemble_dict['max_depth']))
    fout.write('static const int n_features = {};\n'.format(
        ensemble_dict['n_features']))
    fout.write('static const int n_classes = {};\n'.format(
        ensemble_dict['n_classes']))
    fout.write('static const bool unroll = {};\n'.format(
        str(cfg['Pipeline']).lower()))
    fout.write('typedef {} input_t;\n'.format(cfg['Precision_inputs']))
    fout.write('typedef input_t input_arr_t[n_features];\n')
    fout.write('typedef {} score_t;\n'.format(cfg['Precision_score']))
    fout.write('typedef score_t score_arr_t[n_classes];\n')
    # TODO score_arr_t
    fout.write('typedef input_t threshold_t;\n\n')

    tree_fields = ['feature', 'threshold', 'value',
                   'children_left', 'children_right', 'parent']

    fout.write(
        "static const BDT::BDT<n_trees, max_depth, n_classes, input_arr_t, score_t, threshold_t, unroll> bdt = \n")
    fout.write("{ // The struct\n")
    newline = "\t" + str(ensemble_dict['norm']) + ", // The normalisation\n"
    fout.write(newline)
    newline = "\t{"
    if ensemble_dict['n_classes'] > 2:
        for iip, ip in enumerate(ensemble_dict['init_predict']):
            if iip < len(ensemble_dict['init_predict']) - 1:
                newline += '{},'.format(ip)
            else:
                newline += '{}}}, // The init_predict\n'.format(ip)
    else:
        newline += str(ensemble_dict['init_predict'][0]) + '},\n'
    fout.write(newline)
    fout.write("\t// The trees\n")
    # loop over trees
    for itree, trees in enumerate(ensemble_dict['trees']):
        fout.write('\t\t// trees[' + str(itree) + ']\n')
        # loop over classes
        for iclass, tree in enumerate(trees):
            fout.write('\t\t\t{{ // [' + str(iclass) + ']\n')
            # loop over fields
            for ifield, field in enumerate(tree_fields):
                newline = '\t\t\t\t{'
                newline += ','.join(map(str, tree[field]))
                newline += '}'
                if ifield < len(tree_fields) - 1:
                    newline += ','
                newline += '\n'
                fout.write(newline)
            newline = '\t\t\t}'
            if iclass < len(trees) - 1:
                newline += ','
            newline += '\n'
            fout.write(newline)
        newline = '\t\t}'
        if itree < ensemble_dict['n_trees'] - 1:
            newline += ','
        newline += '\n'
        fout.write(newline)
    fout.write('\n};')

    fout.write('\n#endif')
    fout.close()

    #######################
    # myproject.h
    #######################

    f = open(os.path.join(filedir, 'hls-template/firmware/myproject.h'), 'r')
    fout = open(
        '{}/firmware/{}.h'.format(cfg['OutputDir'], cfg['ProjectName']), 'w')

    for line in f.readlines():

        if 'MYPROJECT' in line:
            newline = line.replace(
                'MYPROJECT', format(cfg['ProjectName'].upper()))
        elif 'void myproject(' in line:
            newline = 'void {}(\n'.format(cfg['ProjectName'])
        elif 'hls-fpga-machine-learning insert args' in line:
            newline = '\tinput_arr_t data,\n\tscore_arr_t score'
            if(cfg['OutputTreeScores']):
                newline += ",\n\tscore_t tree_scores[BDT::fn_classes(n_classes) * n_trees]);"
            else:
                newline += ");"
        # Remove some lines

        else:
            newline = line
        fout.write(newline)

    f.close()
    fout.close()

    #######################
    # myproject_test.cpp
    #######################

    f = open(os.path.join(filedir, 'hls-template/myproject_test.cpp'))
    fout = open(
        '{}/{}_test.cpp'.format(cfg['OutputDir'], cfg['ProjectName']), 'w')

    for line in f.readlines():
        indent = ' ' * (len(line) - len(line.lstrip(' ')))
# TODO (Dave) : add support for OutputTreeScores in testbench!
        # Insert numbers
        if 'myproject' in line:
            newline = line.replace('myproject', cfg['ProjectName'])
        elif '//hls-fpga-machine-learning insert data' in line:
            newline = line
            newline += '      std::vector<float>::const_iterator in_begin = in.cbegin();\n'
            newline += '      std::vector<float>::const_iterator in_end;\n'
            newline += '      input_arr_t x;\n'
            newline += '      in_end = in_begin + ({});\n'.format(
                ensemble_dict['n_features'])
            newline += '      std::copy(in_begin, in_end, x);\n'
            newline += '      in_begin = in_end;\n'
            # brace-init zeros the array out because we use std=c++0x
            newline += '      score_arr_t score{};\n'
            newline += '      score_t tree_scores[BDT::fn_classes(n_classes) * n_trees]{};\n'
            # but we can still explicitly zero out if you want
            newline += '      std::fill_n(score, {}, 0.);\n'.format(
                ensemble_dict['n_classes'])
        elif '//hls-fpga-machine-learning insert zero' in line:
            newline = line
            newline += '    input_arr_t x;\n'
            newline += '    std::fill_n(x, {}, 0.);\n'.format(
                ensemble_dict['n_features'])
            newline += '    score_arr_t score{};\n'
            newline += '    score_t tree_scores[BDT::fn_classes(n_classes) * n_trees]{};\n'
            newline += '    std::fill_n(score, {}, 0.);\n'.format(
                ensemble_dict['n_classes'])
        elif '//hls-fpga-machine-learning insert top-level-function' in line:
            newline = line
            if(cfg['OutputTreeScores']):
                fmt_str = '{}(x, score, tree_scores);\n'
            else:
                fmt_str = '{}(x, score);\n'
            top_level = indent + \
                fmt_str.format(cfg['ProjectName'])
            newline += top_level
        elif '//hls-fpga-machine-learning insert predictions' in line:
            newline = line
            newline += indent + \
                'for(int i = 0; i < {}; i++) {{\n'.format(
                    ensemble_dict['n_classes'])
            newline += indent + '  std::cout << pr[i] << " ";\n'
            newline += indent + '}\n'
            newline += indent + 'std::cout << std::endl;\n'
        elif '//hls-fpga-machine-learning insert tb-output' in line:
            newline = line
            newline += indent + \
                'for(int i = 0; i < {}; i++) {{\n'.format(
                    ensemble_dict['n_classes'])
            newline += indent + '  fout << score[i] << " ";\n'
            newline += indent + '}\n'
        elif '//hls-fpga-machine-learning insert output' in line or '//hls-fpga-machine-learning insert quantized' in line:
            newline = line
            newline += indent + \
                'for(int i = 0; i < {}; i++) {{\n'.format(
                    ensemble_dict['n_classes'])
            newline += indent + '  std::cout << score[i] << " ";\n'
            newline += indent + '}\n'
            newline += indent + 'std::cout << std::endl;\n'
        else:
            newline = line
        fout.write(newline)
    # fout.write('#include "BDT.h"\n')
    # fout.write('#include "firmware/parameters.h"\n')
    # fout.write('#include "firmware/{}.h"\n'.format(cfg['ProjectName']))

    #fout.write('int main(){\n')
    #fout.write('\tinput_arr_t x = {{{}}};\n'.format(str([0] * ensemble_dict['n_features'])[1:-1]));
    #fout.write('\tscore_arr_t score;\n')
    #fout.write('\t{}(x, score);\n'.format(cfg['ProjectName']))
    #fout.write('\tfor(int i = 0; i < n_classes; i++){\n')
    #fout.write('\t\tstd::cout << score[i] << ", ";\n\t}\n')
    #fout.write('\tstd::cout << std::endl;\n')
    #fout.write('\treturn 0;\n}')
    # fout.close()

    fout.close()

    #######################
    # build_prj.tcl
    #######################

    bdtdir = os.path.abspath(os.path.join(filedir, "../bdt_utils"))
    relpath = os.path.relpath(bdtdir, start=cfg['OutputDir'])

    f = open(os.path.join(filedir, 'hls-template/build_prj.tcl'), 'r')
    fout = open('{}/build_prj.tcl'.format(cfg['OutputDir']), 'w')

    for line in f.readlines():

        line = line.replace('nnet_utils', relpath)
        line = line.replace('myproject', cfg['ProjectName'])

        # if 'set_top' in line:
        #    line = line.replace('myproject', '{}_decision_function'.format(cfg['ProjectName']))
        if 'set_part {xc7vx690tffg1927-2}' in line:
            line = 'set_part {{{}}}\n'.format(cfg['XilinxPart'])
        elif 'create_clock -period 5 -name default' in line:
            line = 'create_clock -period {} -name default\n'.format(
                cfg['ClockPeriod'])
        # Remove some lines
        elif ('weights' in line) or ('-tb firmware/weights' in line):
            line = ''

        fout.write(line)
    f.close()
    fout.close()


def auto_config():
    config = {'ProjectName': 'my_prj',
              'OutputDir': 'my-conifer-prj',
              'Precision_inputs': 'ap_fixed<18,8>',
              'Precision_score': 'ap_fixed<18,8>',
              'XilinxPart': 'xcvu9p-flgb2104-2L-e',
              'ClockPeriod': '5',
              'Pipeline' : True,
              'Optimized': False,
              'ControlPort': True,
              'OutputTreeScores': True}
    return config


def decision_function(X, config, trees=False):
    np.savetxt('{}/tb_data/tb_input_features.dat'.format(config['OutputDir']),
               X, delimiter=",", fmt='%10f')
    cwd = os.getcwd()
    os.chdir(config['OutputDir'])

    hls_tool = get_hls()
    if hls_tool == None:
        print("No HLS in PATH. Did you source the appropriate Xilinx Toolchain?")
        sys.exit()

    cmd = '{} -f build_prj.tcl "csim=1 synth=0" > predict.log'.format(hls_tool)
    success = os.system(cmd)
    if(success > 0):
        print("'predict' failed, check predict.log")
        sys.exit()
    y = np.loadtxt('tb_data/csim_results.log')
    if trees:
        tree_scores = np.loadtxt('tb_data/csim_tree_results.log')
    os.chdir(cwd)
    if trees:
        return y, tree_scores
    else:
        return y


def sim_compile(config):
    return


def build(config, reset=False, csim=False, synth=True, cosim=False, export=False):
    cwd = os.getcwd()
    os.chdir(config['OutputDir'])

    hls_tool = get_hls()
    if hls_tool == None:
        print("No HLS in PATH. Did you source the appropriate Xilinx Toolchain?")
        sys.exit()

    cmd = '{hls_tool} -f build_prj.tcl "reset={reset} csim={csim} synth={synth} cosim={cosim} export={export}"'\
        .format(hls_tool=hls_tool, reset=reset, csim=csim, synth=synth, cosim=cosim, export=export)
    success = os.system(cmd)
    if(success > 0):
        print("'build' failed")
        sys.exit()
    os.chdir(cwd)
