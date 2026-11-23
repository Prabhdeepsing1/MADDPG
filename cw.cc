#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/opengym-module.h"
//#include "ns3/csma-module.h"
#include "ns3/propagation-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/yans-wifi-channel.h"

#include <fstream>
#include <string>
#include <math.h>
#include <ctime>   //timestampi
#include <iomanip> // put_time
#include <deque>
#include <algorithm>
#include <csignal>
#include "scenario.h"

using namespace std;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE("OpenGym");

void installTrafficGenerator(Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, int port, string offeredLoad, double startTime);
void PopulateARPcache();
void recordHistory();

double envStepTime = 0.01;
double simulationTime = 10; //seconds
double current_time = 0.0;
bool verbose = false;
int end_delay = 0;
bool dry_run = false;

Ptr<FlowMonitor> monitor;
FlowMonitorHelper flowmon;
ofstream outfile ("scratch/linear-mesh/CW_data.csv", fstream::out);

uint32_t CW = 16;


uint32_t history_length = 20;
uint32_t nWifi = 5;

string type = "discrete";
bool non_zero_start = false;
Scenario *wifiScenario;


/*
Define observation space
*/
Ptr<OpenGymSpace> MyGetObservationSpace(void)
{
    float low = 0.0;
    float high = 1.0;
    std::vector<uint32_t> shape = {
        (uint32_t)nWifi,
    };
    std::string dtype = TypeNameGet<float>();
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
    if (verbose)
        NS_LOG_UNCOND("MyGetObservationSpace: " << space);
    return space;
}

/*
Define action space
*/
Ptr<OpenGymSpace> MyGetActionSpace(void)
{
    float low = 0.0;
    float high = 6.0;
    std::vector<uint32_t> shape = {
        (uint32_t)nWifi,
    };
    std::string dtype = TypeNameGet<float>();
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
    if (verbose)
        NS_LOG_UNCOND("MyGetActionSpace: " << space);
    return space;
}

/*
Define extra info. Optional
*/
std::vector<uint32_t> g_rxPktNum;
std::vector<uint32_t> g_txPktNum;
std::vector<uint32_t> g_fairRxSnapshot;
std::vector<uint32_t> g_fairRollingLastRx;
std::deque<std::vector<uint32_t>> g_fairRollingWindows;
const uint32_t FAIRNESS_WINDOW_STEPS = 100;

double jain_index_from_counts(const std::vector<uint32_t>& counts)
{
    double sum = 0.0;
    double sum_squares = 0.0;
    uint32_t positive_stations = 0;

    for (uint32_t i = 0; i < counts.size(); i++) {
        if (counts[i] == 0)
            continue;
        double value = counts[i];
        sum += value;
        sum_squares += value * value;
        positive_stations++;
    }

    if (positive_stations == 0 || sum_squares == 0.0)
        return 1.0;

    return (sum * sum) / ((double)counts.size() * sum_squares);
}

double jain_index(void)
{
    double flowThr;
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    double nominator = 0.0;
    double denominator = 0.0;
    double n=0;
    double station_id = 0;
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        flowThr = i->second.rxBytes;
        flowThr /= wifiScenario->getStationUptime(station_id, current_time);
        if(flowThr>0){
            nominator += flowThr;
            denominator += flowThr*flowThr;
            n++;
        }
        station_id++;
    }
    nominator *= nominator;
    denominator *= n;
    return nominator/denominator;
}

std::string MyGetExtraInfo(void)
{
    static float ticks = 0.0;
    static float lastValue = 0.0;
    
    uint32_t total_rx = 0;
    for (uint32_t i = 0; i < nWifi; i++) {
        total_rx += g_rxPktNum[i];
    }
    
    float obs = total_rx - lastValue;
    lastValue = total_rx;
    ticks += envStepTime;

    float sentMbytes = obs * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024;

    std::vector<uint32_t> per_step_rx(nWifi, 0);
    for (uint32_t i = 0; i < nWifi; i++) {
        per_step_rx[i] = g_rxPktNum[i] - g_fairRollingLastRx[i];
        g_fairRollingLastRx[i] = g_rxPktNum[i];
    }
    g_fairRollingWindows.push_back(per_step_rx);
    if (g_fairRollingWindows.size() > FAIRNESS_WINDOW_STEPS)
        g_fairRollingWindows.pop_front();

    std::vector<uint32_t> rolling_rx(nWifi, 0);
    for (std::deque<std::vector<uint32_t>>::const_iterator step = g_fairRollingWindows.begin();
         step != g_fairRollingWindows.end(); ++step) {
        for (uint32_t i = 0; i < nWifi; i++)
            rolling_rx[i] += (*step)[i];
    }

    std::vector<uint32_t> episode_rx(nWifi, 0);
    for (uint32_t i = 0; i < nWifi; i++)
        episode_rx[i] = g_rxPktNum[i] - g_fairRxSnapshot[i];

    double jainRolling = jain_index_from_counts(rolling_rx);
    double jainEpisode = jain_index_from_counts(episode_rx);

    std::string myInfo = std::to_string(sentMbytes);
    myInfo = myInfo + "|" + to_string(CW) + "|";
    myInfo = myInfo + to_string(wifiScenario->getActiveStationCount(ticks)) + "|";
    myInfo = myInfo + to_string(jain_index()) + "|";
    myInfo = myInfo + to_string(jainRolling) + "|";
    myInfo = myInfo + to_string(jainEpisode) + "|";
    for (uint32_t i = 0; i < nWifi; i++) {
        if (i > 0)
            myInfo = myInfo + ",";
        myInfo = myInfo + to_string(episode_rx[i]);
    }

    if (verbose)
        NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);

    return myInfo;
}

/*
Execute received actions
*/

bool MyExecuteActions(Ptr<OpenGymDataContainer> action)
{
    if (verbose)
        NS_LOG_UNCOND("MyExecuteActions: " << action);

    Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);
    std::vector<float> actionVector = box->GetData();

    uint32_t min_cw = 16;
    uint32_t max_cw = 1024;

    for (uint32_t i = 0; i < nWifi; i++) {
        uint32_t cw_val = 0;
        if (type == "discrete")
        {
            cw_val = pow(2, 4 + actionVector.at(i));
        }
        else if (type == "continuous")
        {
            cw_val = pow(2, actionVector.at(i) + 4);
        }
        else if (type == "direct_continuous")
        {
            cw_val = actionVector.at(i);
        }
        else
        {
            std::cout << "Unsupported agent type!" << endl;
            exit(0);
        }

        cw_val = min(max_cw, max(cw_val, min_cw));

        if (!dry_run) {
            std::string pathMin = "/NodeList/" + std::to_string(i) + "/DeviceList/0/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw";
            std::string pathMax = "/NodeList/" + std::to_string(i) + "/DeviceList/0/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw";
            Config::Set(pathMin, UintegerValue(cw_val));
            Config::Set(pathMax, UintegerValue(cw_val));
        }
        
        if (i == 0) CW = cw_val; 
    }

    outfile << current_time << "," << CW << endl;
    return true;
}

float MyGetReward(void)
{
    static float ticks = 0.0;
    static uint32_t last_packets = 0;
    static float last_reward = 0.0;
    ticks += envStepTime;

    uint32_t total_rx = 0;
    for (uint32_t i = 0; i < nWifi; i++) {
        total_rx += g_rxPktNum[i];
    }

    float res = total_rx - last_packets;
    float reward = res * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024 / (nWifi * 150.0 * envStepTime) * 10;  // normalize by active station count

    last_packets = total_rx;

    if (ticks <= 2 * envStepTime)
        return 0.0;

    if (verbose)
        NS_LOG_UNCOND("MyGetReward: " << reward);

    if(reward>1.0f || reward<0.0f)
        reward = last_reward;
    last_reward = reward;
    return last_reward;
}

/*
Collect observations
*/

Ptr<OpenGymDataContainer> MyGetObservation()
{
    static std::vector<uint32_t> last_rx(nWifi, 0);
    static std::vector<uint32_t> last_tx(nWifi, 0);

    std::vector<uint32_t> shape = {
        (uint32_t)nWifi,
    };
    Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(shape);

    for (uint32_t i = 0; i < nWifi; i++)
    {
        float received = g_rxPktNum[i] - last_rx[i];
        float sent = g_txPktNum[i] - last_tx[i];
        
        float ratio = 0.0;
        if (sent > 0) {
            float errs = sent - received;
            ratio = errs / sent;
        }
        
        if (ratio >= -100 && ratio <= 100)
            box->AddValue(ratio);
        else
            box->AddValue(0);
            
        last_rx[i] = g_rxPktNum[i];
        last_tx[i] = g_txPktNum[i];
    }

    if (verbose)
        NS_LOG_UNCOND("MyGetObservation: " << box);
    return box;
}

bool MyGetGameOver(void)
{

    return false;
}

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGymInterface)
{
    Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGymInterface);
    openGymInterface->NotifyCurrentState();
}

void recordHistory()
{
    static uint32_t calls = 0;
    calls++;
    current_time += envStepTime;

    if (calls < history_length && non_zero_start)
    {
        Simulator::Schedule(Seconds(envStepTime), &recordHistory);
    }
    else if (calls == history_length && non_zero_start)
    {
        for (uint32_t i = 0; i < nWifi; i++) {
            g_rxPktNum[i] = 0;
            g_txPktNum[i] = 0;
        }
    }
}

void packetReceived(std::string context, Ptr<const Packet> packet)
{
    NS_LOG_DEBUG("Client received a packet of " << packet->GetSize() << " bytes");
    size_t first = context.find("/NodeList/");
    if (first != std::string::npos) {
        size_t second = context.find("/", first + 10);
        uint32_t id = 0;
        if (second != std::string::npos) {
            id = std::stoi(context.substr(first + 10, second - (first + 10)));
        } else {
            id = std::stoi(context.substr(first + 10));
        }
        if (id < nWifi) {
            g_rxPktNum[id]++;
        }
    }
}

void packetSent(std::string context, Ptr<const Packet> packet)
{
    size_t first = context.find("/NodeList/");
    if (first != std::string::npos) {
        size_t second = context.find("/", first + 10);
        uint32_t id = 0;
        if (second != std::string::npos) {
            id = std::stoi(context.substr(first + 10, second - (first + 10)));
        } else {
            id = std::stoi(context.substr(first + 10));
        }
        if (id < nWifi) {
            g_txPktNum[id]++;
        }
    }
}

void set_phy(int nWifi, int guardInterval, NodeContainer &wifiStaNode, NodeContainer &wifiApNode, YansWifiPhyHelper &phy)
{
    Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel>();
    lossModel->SetDefaultLoss(50);

    wifiStaNode.Create(nWifi);
    wifiApNode.Create(1);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    Ptr<YansWifiChannel> chan = channel.Create();
    chan->SetPropagationLossModel(lossModel);
    chan->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

    phy = YansWifiPhyHelper::Default();
    phy.SetChannel(chan);

    // Set guard interval
    phy.Set("GuardInterval", TimeValue(NanoSeconds(guardInterval)));
}

void set_nodes(int channelWidth, int rng, int32_t simSeed, NodeContainer wifiStaNode, NodeContainer wifiApNode, YansWifiPhyHelper phy, WifiMacHelper mac, WifiHelper wifi, NetDeviceContainer &apDevice)
{
    // Set the access point details
    Ssid ssid = Ssid("ns3-80211ax");

    mac.SetType("ns3::StaWifiMac",
                "Ssid", SsidValue(ssid),
                "ActiveProbing", BooleanValue(false),
                "BE_MaxAmpduSize", UintegerValue(0));

    NetDeviceContainer staDevice;
    staDevice = wifi.Install(phy, mac, wifiStaNode);

    mac.SetType("ns3::ApWifiMac",
                "EnableBeaconJitter", BooleanValue(false),
                "Ssid", SsidValue(ssid));

    apDevice = wifi.Install(phy, mac, wifiApNode);

    // Set channel width
    Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelWidth", UintegerValue(channelWidth));

    // mobility.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(1.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNode);
    /* Internet stack*/
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNode);

    //Random
    if(simSeed!=-1)
        RngSeedManager::SetSeed(simSeed);
    RngSeedManager::SetRun(rng);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staNodeInterface;
    Ipv4InterfaceContainer apNodeInterface;

    staNodeInterface = address.Assign(staDevice);
    apNodeInterface = address.Assign(apDevice);

    if (!dry_run)
    {
        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw", UintegerValue(CW));
        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw", UintegerValue(CW));
    }
    else
    {
        NS_LOG_UNCOND("Default CW");
        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw", UintegerValue(16));
        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw", UintegerValue(1024));
    }
}

void set_sim(bool tracing, bool dry_run, int warmup, uint32_t openGymPort, YansWifiPhyHelper phy, NetDeviceContainer apDevice, int end_delay, Ptr<FlowMonitor> &monitor, FlowMonitorHelper &flowmon)
{
    monitor = flowmon.InstallAll();
    monitor->SetAttribute("StartTime", TimeValue(Seconds(warmup)));

    if (tracing)
    {
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        phy.EnablePcap("cw", apDevice.Get(0));
    }

    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(openGymPort);
    openGymInterface->SetGetActionSpaceCb(MakeCallback(&MyGetActionSpace));
    openGymInterface->SetGetObservationSpaceCb(MakeCallback(&MyGetObservationSpace));
    openGymInterface->SetGetGameOverCb(MakeCallback(&MyGetGameOver));
    openGymInterface->SetGetObservationCb(MakeCallback(&MyGetObservation));
    openGymInterface->SetGetRewardCb(MakeCallback(&MyGetReward));
    openGymInterface->SetGetExtraInfoCb(MakeCallback(&MyGetExtraInfo));
    openGymInterface->SetExecuteActionsCb(MakeCallback(&MyExecuteActions));


    if (non_zero_start)
    {
        Simulator::Schedule(Seconds(1.0), &recordHistory);
        Simulator::Schedule(Seconds(envStepTime * history_length + 1.0), &ScheduleNextStateRead, envStepTime, openGymInterface);
    }
    else
        Simulator::Schedule(Seconds(1.0), &ScheduleNextStateRead, envStepTime, openGymInterface);


    Simulator::Stop(Seconds(simulationTime + end_delay + 1.0 + envStepTime*(history_length+1)));

    NS_LOG_UNCOND("Simulation started");
    Simulator::Run();
}

void signalHandler(int signum)
{
    cout << "Interrupt signal " << signum << " received.\n";
    exit(signum);
}

int main(int argc, char *argv[])
{
    bool tracing = false;
    bool useRts = false;
    int mcs = 11;
    int channelWidth = 20;
    int guardInterval = 800;
    string offeredLoad = "150";
    int port = 1025;
    string outputCsv = "cw.csv";
    string scenario = "basic";
    dry_run = false;

    int rng = 1;
    int warmup = 1;

    uint32_t openGymPort = 5555;
    int32_t simSeed = -1;

    signal(SIGTERM, signalHandler);
    outfile << "SimulationTime,CW" << endl;

    CommandLine cmd;
    cmd.AddValue("openGymPort", "Specify port number. Default: 5555", openGymPort);
    cmd.AddValue("CW", "Value of Contention Window", CW);
    cmd.AddValue("historyLength", "Length of history window", history_length);
    cmd.AddValue("nWifi", "Number of wifi 802.11ax STA devices", nWifi);
    cmd.AddValue("verbose", "Tell echo applications to log if true", verbose);
    cmd.AddValue("tracing", "Enable pcap tracing", tracing);
    cmd.AddValue("rng", "Number of RngRun", rng);
    cmd.AddValue("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
    cmd.AddValue("envStepTime", "Step time in seconds. Default: 0.1s", envStepTime);
    cmd.AddValue("agentType", "Type of agent actions: discrete, continuous", type);
    cmd.AddValue("nonZeroStart", "Start only after history buffer is filled", non_zero_start);
    cmd.AddValue("scenario", "Scenario for analysis: basic, convergence, reaction", scenario);
    cmd.AddValue("dryRun", "Execute scenario with BEB and no agent interaction", dry_run);
    cmd.AddValue("seed", "Random seed", simSeed);

    cmd.Parse(argc, argv);

    g_rxPktNum.resize(nWifi, 0);
    g_txPktNum.resize(nWifi, 0);
    g_fairRxSnapshot.assign(nWifi, 0);
    g_fairRollingLastRx.assign(nWifi, 0);
    g_fairRollingWindows.clear();


    NS_LOG_UNCOND("Ns3Env parameters:");
    NS_LOG_UNCOND("--nWifi: " << nWifi);
    NS_LOG_UNCOND("--simulationTime: " << simulationTime);
    NS_LOG_UNCOND("--openGymPort: " << openGymPort);
    NS_LOG_UNCOND("--envStepTime: " << envStepTime);
    NS_LOG_UNCOND("--seed: " << simSeed);
    NS_LOG_UNCOND("--agentType: " << type);
    NS_LOG_UNCOND("--scenario: " << scenario);
    NS_LOG_UNCOND("--dryRun: " << dry_run);

    if (verbose)
    {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }

    if (useRts)
    {
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue("0"));
    }

    NodeContainer wifiStaNode;
    NodeContainer wifiApNode;
    YansWifiPhyHelper phy;
    set_phy(nWifi, guardInterval, wifiStaNode, wifiApNode, phy);

    WifiMacHelper mac;
    WifiHelper wifi;

    wifi.SetStandard(WIFI_PHY_STANDARD_80211ax_5GHZ);

    std::ostringstream oss;
    oss << "HeMcs" << mcs;
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue(oss.str()),
                                 "ControlMode", StringValue(oss.str()));


    NetDeviceContainer apDevice;
    set_nodes(channelWidth, rng, simSeed, wifiStaNode, wifiApNode, phy, mac, wifi, apDevice);

    ScenarioFactory helper = ScenarioFactory(nWifi, wifiStaNode, wifiApNode, port, offeredLoad, history_length);
    wifiScenario = helper.getScenario(scenario);


    if (non_zero_start)
        end_delay = envStepTime * history_length + 1.0;
    else
        end_delay = 0.0;


    wifiScenario->installScenario(simulationTime + end_delay + envStepTime, envStepTime, MakeCallback(&packetReceived));

    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin", MakeCallback(&packetSent));

    wifiScenario->PopulateARPcache();
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    set_sim(tracing, dry_run, warmup, openGymPort, phy, apDevice, end_delay, monitor, flowmon);

    double flowThr;
    uint32_t final_total_rx = 0;
    for (uint32_t i = 0; i < nWifi; i++) {
        final_total_rx += g_rxPktNum[i];
    }
    uint32_t episode_rx_min = g_rxPktNum.empty() ? 0 : g_rxPktNum[0] - g_fairRxSnapshot[0];
    uint32_t episode_rx_max = episode_rx_min;
    for (uint32_t i = 0; i < nWifi; i++) {
        uint32_t episode_rx = g_rxPktNum[i] - g_fairRxSnapshot[i];
        episode_rx_min = min(episode_rx_min, episode_rx);
        episode_rx_max = max(episode_rx_max, episode_rx);
    }
    float res =  final_total_rx * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024;
    printf("Sent mbytes: %.2f\tThroughput: %.3f", res, res/simulationTime);
    printf("\tEpisodeRxPacketsMin: %u\tEpisodeRxPacketsMax: %u", episode_rx_min, episode_rx_max);
    ofstream myfile;
    myfile.open(outputCsv, ios::app);


    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        auto time = std::time(nullptr); 
        auto tm = *std::localtime(&time);
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        flowThr = i->second.rxBytes * 8.0 / simulationTime / 1000 / 1000;
        NS_LOG_UNCOND("Flow " << i->first << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\tThroughput: " << flowThr << " Mbps\tTime: " << i->second.timeLastRxPacket.GetSeconds() - i->second.timeFirstTxPacket.GetSeconds() << " s\tRx packets " << i->second.rxPackets);
        myfile << std::put_time(&tm, "%Y-%m-%d %H:%M") << "," << CW << "," << nWifi << "," << RngSeedManager::GetRun() << "," << t.sourceAddress << "," << t.destinationAddress << "," << flowThr;
        myfile << std::endl;
    }
    myfile.close();

    Simulator::Destroy();
    NS_LOG_UNCOND("Packets registered by handler: " << final_total_rx << " Packets" << endl);

    return 0;
}
