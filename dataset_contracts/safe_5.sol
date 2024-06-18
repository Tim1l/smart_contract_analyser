// simple_list.sol
pragma solidity ^0.4.15;

contract SimpleList {
    string[] public list;

    function addItem(string item) public {
        list.push(item);
    }

    function getItem(uint index) public constant returns (string) {
        require(index < list.length);
        return list[index];
    }

    function getListLength() public constant returns (uint) {
        return list.length;
    }
}