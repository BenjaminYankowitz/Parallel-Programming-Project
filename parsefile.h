
#ifndef ParseFile__H__
#include <iostream>
#include <algorithm>
#include <charconv>
#include <array>
#include <vector>
#include "mpi.h"

inline void printErrorStatus(int errorPrint){
    switch (errorPrint) {
        case MPI_SUCCESS:
        break;
        case MPI_ERR_COMM:
        std::cout << "Invalid communicator";
        break;
        case MPI_ERR_TYPE:
        std::cout << "Invalid datatype argument";
        break;
        case MPI_ERR_COUNT:
        std::cout << "Invalid count argument";
        break;
        case MPI_ERR_TAG:
        std::cout << "Invalid tag argument";
        break;
        case MPI_ERR_RANK:
        std::cout << "Invalid source or destination rank";
        break;
        default:
        std::cout << "Error error number";
        break;
    }
}

inline std::vector<std::vector<int>> readFile(const char* fileName){
    MPI_File fh;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_File_open(MPI_COMM_WORLD, fileName,
        MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    const MPI_Offset fileSize = [&]{
        MPI_Offset fileSize;
        MPI_File_get_size(fh,&fileSize);
        return fileSize;
    }();
    const MPI_Offset groupSize=fileSize/world_size;
    const MPI_Offset startOffset = groupSize*world_rank;
    constexpr MPI_Offset overlap = 25;
    const MPI_Offset charsToRead = [&]{
        if(world_rank+1==world_size){
            return fileSize-startOffset;
        } else {
            return groupSize+overlap;
        }
    }();   
    std::vector<char> buffer(charsToRead);
    MPI_Status status;
    MPI_File_read_at(fh,startOffset,buffer.data(),charsToRead,MPI_CHAR,&status);
    int count;
    MPI_Get_count(&status,MPI_CHAR,&count);
    if(count!=charsToRead){
        std::cerr << "Read failed, tried to read " << charsToRead << " chars, but only read " << count << " chars\n";
    }
    std::size_t loc = 0;
    const auto getNextNewLine = [&buffer,&loc,world_rank,world_size](){
        std::size_t old_loc = loc;
        if(world_rank+1!=world_size && loc+overlap>buffer.size()){
            loc = buffer.size();
            return std::string_view();
        }
        for(std::size_t i = loc; i < buffer.size(); i++){
            if(buffer[i]=='\n'){
                loc = i+1;
                return std::string_view(buffer.data()+old_loc,i-old_loc);
            }
        }
        loc = buffer.size();
        if(world_rank+1==world_size){
            return std::string_view(buffer.data()+old_loc,buffer.size()-old_loc);
        }
        return std::string_view();
    };
    getNextNewLine();
    std::vector<std::pair<int,std::vector<int>>> readData;
    for(std::string_view line = getNextNewLine();line.size()!=0; line = getNextNewLine()){
        if(line[0]=='#'){
            continue;
        }
        int to;
        int from;
        std::size_t tabP = line.find_first_of('\t');
        std::from_chars(line.data(),line.data()+tabP+1,from);
        std::from_chars(line.data()+tabP+1,line.data()+line.size(),to);
        if(!readData.empty()&&readData.back().first==from){
            readData.back().second.push_back(to);
        } else {
            readData.push_back(std::pair<int,std::vector<int>>(from,{to}));
        }
    }
    MPI_File_close(&fh);
    std::vector<int> startVals(world_size,-1);
    startVals[world_rank] = 0;
    std::vector<std::vector<std::vector<int>>> splitValues(world_size);
    for(std::pair<int,std::vector<int>>& line : readData){
        int owner = line.first%world_size;
        int index = line.first/world_size;
        if(startVals[owner]==-1){
            startVals[owner] = index;
        }
        if(index<startVals[owner]){
            std::cerr << "bad\n";
            exit(1);
        }
        std::size_t modIndex = index - startVals[owner];
        if(splitValues[owner].size()<=modIndex){
            splitValues[owner].resize(modIndex+1);
        }
        splitValues[owner][modIndex] = std::move(line.second);
    }
    std::vector<std::vector<int>> ret = std::move(splitValues[world_rank]);
    const auto doRecive = [&](int partner){
        std::array<int,2> dataToRecive = {0};
        MPI_Status status;
        MPI_Recv(dataToRecive.data(),dataToRecive.size(),MPI_INT,partner,0,MPI_COMM_WORLD,&status);
        int count;
        MPI_Get_count(&status,MPI_INT,&count);
        if(static_cast<unsigned>(count)!=dataToRecive.size()){
            std::cout << "failed to read size of vector, instead of " << dataToRecive.size() << ", " << count << " were read instead\n";
            printErrorStatus(status.MPI_ERROR);
            std::cout << '\n';
        }
        if(dataToRecive[0]==-1){
            return;
        }
        std::size_t theirStartVal = dataToRecive[0];
        if(theirStartVal+1==0){}
        std::size_t numReciv = dataToRecive[1];
        if(ret.size()<theirStartVal+numReciv){
            ret.resize(theirStartVal+numReciv);
        }
        for(std::size_t i = 0; i < numReciv; i++){
            long vecSize;
            MPI_Recv(&vecSize,1,MPI_LONG,partner,0,MPI_COMM_WORLD,&status);
            MPI_Get_count(&status,MPI_LONG,&count);
            if(count!=1){
                std::cout << "failed to read size of vector, instead of 1, " << count << " were read instead\n";
                printErrorStatus(status.MPI_ERROR);
                std::cout << '\n';
            }
            std::size_t oldelemNum = ret[theirStartVal+i].size();
            ret[theirStartVal+i].resize(oldelemNum+vecSize);
            MPI_Recv(ret[theirStartVal+i].data()+oldelemNum, vecSize, MPI_INT, partner, 0, MPI_COMM_WORLD,&status);
            MPI_Get_count(&status,MPI_INT,&count);
            if(count!=vecSize){
                std::cout << "failed to read " << vecSize << " elems only read " << status._ucount << "\n";
                printErrorStatus(status.MPI_ERROR);
                std::cout << '\n';
            }
        }        
    };
    const auto doSend = [&](int partner){
        std::array<int,2> dataToSend = {startVals[partner], (int)splitValues[partner].size()};
        MPI_Send(dataToSend.data(),dataToSend.size(),MPI_INT,partner,0,MPI_COMM_WORLD);
        if(startVals[partner]==-1){
            return;
        }
        for(std::vector<int>& vec : splitValues[partner]){
            const long vecSize = vec.size();
            MPI_Send(&vecSize,1,MPI_LONG,partner,0,MPI_COMM_WORLD);
            MPI_Send(vec.data(), vecSize, MPI_INT, partner, 0, MPI_COMM_WORLD);
        }
    };
    for(int rankahead = 1; rankahead <= world_size/2; rankahead++){
        int partnerF = (world_rank+rankahead)%world_size;
        int partnerB = (world_size+world_rank-rankahead)%world_size;
        const auto cmpTasks = [rankahead,world_rank,partnerF,partnerB](){
            if(partnerF<world_rank)
                return false;
            if(partnerB>world_rank)
                return true;
            return (partnerB/rankahead)%2==0;
        };
        int partner1 = partnerF;
        int partner2 = partnerB;
        if(cmpTasks()){
            std::swap(partner1,partner2);
        }
        const auto doSwap = [&](int partner){
            if(world_rank<partner){
                doSend(partner);
                doRecive(partner);
            } else {
                doRecive(partner);
                doSend(partner);
            }
        };
        doSwap(partner1);
        if(partner1!=partner2){
            doSwap(partner2);
        }
    }
    for(std::vector<int>& vec : ret){
        std::sort(vec.begin(),vec.end());
    }

    return ret;
}

#endif