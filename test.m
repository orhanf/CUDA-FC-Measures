
%% caution for zero-based indexing in C
[status, result] = system('queryDistMat.exe -q 0 -c 4096 -n 8142 -d');

fid = fopen('queryRowResult_0.bin');
fread(fid,4,'single');
fclose(fid);


%%
% CUDA_FC_MEASURES.exe -i valence_data2.txt -n 82926 -t 984 -d 
resultMat = zeros(8142,8142);
stats = zeros(8142,1);

for i=0:8141

    resultVec = queryDistMat(82926, 4096, i, 'C:\FMRI\fc_prj\outputs\fc_valence\class_1\', 0);
    resultMat(i+1,:) = resultVec;
    stats(i+1) = sum(resultVec>0.9);
end    
    

%% valence classwise

resultVec = queryDistMat(82926, 4096, queryRow, 'C:\FMRI\fc_prj\outputs\fc_valence\class_1\', 0);

%% pi_brown all

queryRow = 1;
resultVec = queryDistMat(8142, 4096, queryRow, 'C:\FMRI\fc_prj\outputs\fc_pi_brown\class_all\', 0);

