// decentralized_voting.sol
pragma solidity ^0.4.15;

contract DecentralizedVoting {
    struct Proposal {
        string description;
        uint voteCount;
    }

    Proposal[] public proposals;
    mapping (address => bool) public voters;

    function addProposal(string description) public {
        proposals.push(Proposal({
            description: description,
            voteCount: 0
        }));
    }

    function vote(uint proposalIndex) public {
        require(!voters[msg.sender]);
        require(proposalIndex < proposals.length);
        
        proposals[proposalIndex].voteCount += 1;
        voters[msg.sender] = true;
    }

    function getProposal(uint index) public constant returns (string description, uint voteCount) {
        require(index < proposals.length);
        Proposal storage proposal = proposals[index];
        return (proposal.description, proposal.voteCount);
    }

    function getProposalCount() public constant returns (uint) {
        return proposals.length;
    }
}