#include<map>
#include<vector>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <any>
#include "dataStructures.h"

using namespace std;

struct CSV_Line{
	std::map<int, std::string> lineMap;
};

class CSV_Writer
{
private:
    //Column name - column value type
    map<int, string, std::less<int>> _csvCols;
    vector<CSV_Line> _contents;
    string _separator;
    void AddContentsToStream(map<int, string> lineContents, fstream& currentStream);
public:
    CSV_Writer(map<int, string> colsNames, string separator);
    ~CSV_Writer();
    void AddLine(const CSV_Line& line);
    bool SaveFile(string filePath);
};

CSV_Writer::CSV_Writer(map<int, string> cols, string separator = ",")
{
    _separator = separator;
    for(const auto& col : cols)
    {
        _csvCols.insert(col);
    }

}

CSV_Writer::~CSV_Writer()
{
}

void CSV_Writer::AddLine(const CSV_Line& line)
{
    _contents.push_back(line);
}

bool CSV_Writer::SaveFile(string filePath)
{
    try
    {
        bool existed = false;
        ifstream file(filePath);
        if(file)
        {
            existed = true;
            file.close();
        }
        fstream fileStream (filePath, std::ios_base::app | std::ios_base::out);
        string lineTxt = "";
        if(existed)
        {
            for(const CSV_Line& line : _contents)
            {
                AddContentsToStream(line.lineMap, fileStream);
            }
        }
        else
        {
            for(const auto& col : _csvCols)
            {
                lineTxt += col.second + _separator;
            }
            lineTxt+="\n";
            fileStream<<lineTxt;
            for(const CSV_Line& line : _contents)
            {
                AddContentsToStream(line.lineMap, fileStream);
            }
        }
        lineTxt = "";
        for(const auto& col : _csvCols)
        {
            lineTxt += " " + _separator;
        }
        lineTxt+="\n";
        fileStream<<lineTxt;
        fileStream.close();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return false;
    }
}

void CSV_Writer::AddContentsToStream(map<int, string> lineContents, fstream& currentStream)
{
    map<int, string, std::less<int>> orderedMap;
    for(const auto& line : lineContents)
    {
        orderedMap.insert(make_pair(line.first, line.second));
    }
    string lineTxt = "";
    for(const auto& entry : orderedMap)
    {
        lineTxt += entry.second + _separator;
    }
    lineTxt += "\n";
    currentStream<<lineTxt;
}
